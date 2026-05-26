"""
credits.py — Net-Neutral AI
SQLite credit ledger — tracks compute contributions and rewards.

Spec reference: Backend Schema Document Section 3, Section 4

Tables managed:
    credits : one row per client per round
    rounds  : one row per completed round

All database operations are in this file.
server.py calls these functions — it never writes SQL directly.

Database location: coordinator/database.db (auto-created on first run)
"""

import os
import sqlite3
import datetime
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# ── Database path ─────────────────────────────────────────────────────────────
COORDINATOR_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH         = os.path.join(COORDINATOR_DIR, "database.db")

# ── Credit formula ────────────────────────────────────────────────────────────
# Spec: points = floor(samples_trained / 5)
def compute_points(samples_trained: int) -> int:
    return samples_trained // 5


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class CreditRecord:
    """One row from the credits table."""
    id:              int
    client_id:       str
    round:           int
    samples_trained: int
    time_seconds:    float
    points_earned:   int
    timestamp:       str


@dataclass
class RoundRecord:
    """One row from the rounds table."""
    id:                int
    round_number:      int
    started_at:        str
    completed_at:      str
    clients_submitted: int
    global_accuracy:   float
    accuracy_delta:    float


@dataclass
class LeaderboardEntry:
    """One row from the leaderboard query."""
    rank:               int
    client_id:          str
    total_points:       int
    total_samples:      int
    rounds_participated: int


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> None:
    """
    Create database and tables if they don't exist.
    Safe to call multiple times — uses IF NOT EXISTS.
    Called once by server.py on startup.

    Args:
        db_path : path to SQLite database file
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS credits (
            id               INTEGER  PRIMARY KEY AUTOINCREMENT,
            client_id        TEXT     NOT NULL,
            round            INTEGER  NOT NULL,
            samples_trained  INTEGER  NOT NULL DEFAULT 0,
            time_seconds     REAL     NOT NULL DEFAULT 0.0,
            points_earned    INTEGER  NOT NULL DEFAULT 0,
            timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS rounds (
            id                 INTEGER  PRIMARY KEY AUTOINCREMENT,
            round_number       INTEGER  NOT NULL UNIQUE,
            started_at         DATETIME NOT NULL,
            completed_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
            clients_submitted  INTEGER  NOT NULL DEFAULT 0,
            global_accuracy    REAL     NOT NULL DEFAULT 0.0,
            accuracy_delta     REAL     DEFAULT 0.0
        );

        CREATE INDEX IF NOT EXISTS idx_credits_client
            ON credits (client_id);

        CREATE INDEX IF NOT EXISTS idx_credits_round
            ON credits (round);
    """)

    conn.commit()
    conn.close()
    print(f"[Credits] ✓  Database initialised at {db_path}")


def reset_db(db_path: str = DB_PATH) -> None:
    """
    Drop all data and recreate tables.
    Call this between demo runs to start fresh.
    WARNING: deletes all credits and round history.

    Args:
        db_path : path to SQLite database file
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript("""
        DROP TABLE IF EXISTS credits;
        DROP TABLE IF EXISTS rounds;
        DROP INDEX IF EXISTS idx_credits_client;
        DROP INDEX IF EXISTS idx_credits_round;
    """)
    conn.commit()
    conn.close()
    init_db(db_path)
    print(f"[Credits] ✓  Database reset complete")


# ─────────────────────────────────────────────────────────────────────────────
# WRITE OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def log_credit(
    client_id:       str,
    round_num:       int,
    samples_trained: int,
    time_seconds:    float,
    db_path:         str = DB_PATH,
) -> int:
    """
    Insert one credit record after a client submits weights.
    Points are computed here — never trusted from client.

    Args:
        client_id       : e.g. 'client_A'
        round_num       : current round number
        samples_trained : number of samples client trained on
        time_seconds    : wall-clock training time
        db_path         : path to database

    Returns:
        points_earned : computed credit points for this submission
    """
    # Guard: reject duplicate submission for same client + round
    if _submission_exists(client_id, round_num, db_path):
        print(f"[Credits] ⚠  Duplicate submission ignored: {client_id} round {round_num}")
        return 0

    points = compute_points(samples_trained)

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO credits (client_id, round, samples_trained, time_seconds, points_earned)
        VALUES (?, ?, ?, ?, ?)
        """,
        (client_id, round_num, samples_trained, round(time_seconds, 2), points)
    )
    conn.commit()
    conn.close()

    print(f"[Credits] ✓  Logged: {client_id} | round {round_num} | "
          f"{samples_trained} samples | {points} pts")
    return points


def log_round(
    round_number:      int,
    started_at:        datetime.datetime,
    clients_submitted: int,
    global_accuracy:   float,
    db_path:           str = DB_PATH,
) -> float:
    """
    Insert one round record after FedAvg and evaluation complete.
    Computes accuracy_delta vs previous round automatically.

    Args:
        round_number      : 1-indexed round number
        started_at        : datetime when operator typed start_round
        clients_submitted : how many clients submitted this round
        global_accuracy   : model accuracy after FedAvg (0.0–1.0)
        db_path           : path to database

    Returns:
        accuracy_delta : change vs previous round (negative if regression)
    """
    # Get previous round accuracy for delta calculation
    prev_accuracy = _get_previous_accuracy(round_number, db_path)
    delta         = global_accuracy - prev_accuracy

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO rounds
            (round_number, started_at, clients_submitted, global_accuracy, accuracy_delta)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            round_number,
            started_at.strftime("%Y-%m-%d %H:%M:%S"),
            clients_submitted,
            round(global_accuracy, 6),
            round(delta, 6),
        )
    )
    conn.commit()
    conn.close()

    delta_str = f"{'+' if delta >= 0 else ''}{delta*100:.2f}%"
    print(f"[Credits] ✓  Round {round_number} logged | "
          f"accuracy: {global_accuracy*100:.2f}% ({delta_str}) | "
          f"{clients_submitted} clients")
    return delta


# ─────────────────────────────────────────────────────────────────────────────
# READ OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_leaderboard(db_path: str = DB_PATH) -> List[LeaderboardEntry]:
    """
    Get cumulative credit leaderboard — all clients sorted by total points.

    Returns:
        List of LeaderboardEntry sorted descending by total_points
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            client_id,
            SUM(points_earned)   AS total_points,
            SUM(samples_trained) AS total_samples,
            COUNT(*)             AS rounds_participated
        FROM credits
        GROUP BY client_id
        ORDER BY total_points DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    return [
        LeaderboardEntry(
            rank                = i + 1,
            client_id           = row[0],
            total_points        = row[1],
            total_samples       = row[2],
            rounds_participated = row[3],
        )
        for i, row in enumerate(rows)
    ]


def get_accuracy_history(db_path: str = DB_PATH) -> List[Tuple[int, float]]:
    """
    Get per-round accuracy history for the final summary printout.

    Returns:
        List of (round_number, global_accuracy) tuples ordered by round
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT round_number, global_accuracy
        FROM rounds
        ORDER BY round_number ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [(row[0], row[1]) for row in rows]


def get_submitted_clients(round_num: int, db_path: str = DB_PATH) -> List[str]:
    """
    Get list of client_ids that have submitted for a given round.
    Used by server.py to check if all clients have submitted.

    Args:
        round_num : round number to check

    Returns:
        List of client_ids who submitted this round
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT client_id FROM credits WHERE round = ?",
        (round_num,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def get_round_credits(
    round_num: int,
    db_path:   str = DB_PATH,
) -> Dict[str, int]:
    """
    Get credits earned by each client in a specific round.
    Used by server.py to include in the round summary terminal output.

    Args:
        round_num : round number

    Returns:
        Dict mapping client_id → points_earned this round
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT client_id, points_earned FROM credits WHERE round = ?",
        (round_num,)
    )
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def get_client_total_credits(
    client_id: str,
    db_path:   str = DB_PATH,
) -> int:
    """
    Get cumulative total credits for one client across all rounds.
    Returned to client after each submission so they can display it.

    Args:
        client_id : e.g. 'client_A'

    Returns:
        Total points earned across all rounds
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COALESCE(SUM(points_earned), 0) FROM credits WHERE client_id = ?",
        (client_id,)
    )
    total = cursor.fetchone()[0]
    conn.close()
    return total


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

PRINT_WIDTH = 60

def _line(char="─"): return char * PRINT_WIDTH

def format_leaderboard(entries: List[LeaderboardEntry], round_num: int) -> str:
    """
    Format leaderboard for terminal output after each round.
    Called by server.py to print after FedAvg completes.
    """
    lines = []
    lines.append(_line("─"))
    lines.append(f"  Leaderboard after Round {round_num}")
    lines.append(_line("─"))
    lines.append(f"  {'Rank':<6}{'Client':<14}{'Points':>10}{'Samples':>12}{'Rounds':>8}")
    lines.append(_line("─"))
    for e in entries:
        lines.append(
            f"  {e.rank:<6}{e.client_id:<14}"
            f"{e.total_points:>10,}"
            f"{e.total_samples:>12,}"
            f"{e.rounds_participated:>8}"
        )
    lines.append(_line("─"))
    return "\n".join(lines)


def format_final_leaderboard(entries: List[LeaderboardEntry]) -> str:
    """
    Format the final leaderboard for the end-of-session summary.
    More detailed than the per-round version.
    """
    lines = []
    lines.append(_line("═"))
    lines.append("  FINAL LEADERBOARD")
    lines.append(_line("═"))
    lines.append(
        f"  {'Rank':<6}{'Client':<14}{'Total Pts':>12}"
        f"{'Total Samples':>15}{'Rounds':>8}"
    )
    lines.append(_line("─"))
    for e in entries:
        medal = ["Gold", "Silver", "Bronze"][e.rank - 1] if e.rank <= 3 else "  "
        lines.append(
            f"  {medal} {e.rank:<4}{e.client_id:<14}"
            f"{e.total_points:>12,}"
            f"{e.total_samples:>15,}"
            f"{e.rounds_participated:>8}"
        )
    lines.append(_line("═"))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _submission_exists(
    client_id: str,
    round_num: int,
    db_path:   str,
) -> bool:
    """Check if a credit record already exists for this client + round."""
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM credits WHERE client_id = ? AND round = ?",
        (client_id, round_num)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def _get_previous_accuracy(round_number: int, db_path: str) -> float:
    """
    Get the global_accuracy of the previous round.
    Returns 0.0 for round 1 (no previous round exists).
    """
    if round_number <= 1:
        return 0.0
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT global_accuracy FROM rounds WHERE round_number = ?",
        (round_number - 1,)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    PRINT_WIDTH = 60
    print("=" * PRINT_WIDTH)
    print("  credits.py sanity check")
    print("=" * PRINT_WIDTH)

    # Use a temp database so we don't pollute the real one
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db_path = tmp_db.name
    tmp_db.close()

    try:
        # ── Test 1: init ──────────────────────────────────────────────────────
        print("\n[Test 1] Database initialisation")
        init_db(tmp_db_path)
        print("  ✓  Tables created")

        # ── Test 2: log credits ───────────────────────────────────────────────
        print("\n[Test 2] Logging credits for 3 clients, 2 rounds")
        started = datetime.datetime(2026, 6, 1, 10, 0, 0)

        for round_num in [1, 2]:
            for client_id, samples in [("client_A", 5000), ("client_B", 5000), ("client_C", 5000)]:
                pts = log_credit(client_id, round_num, samples, 30.0, tmp_db_path)
                assert pts == 1000, f"Expected 1000 pts, got {pts}"

        print("  ✓  All credit records inserted correctly")

        # ── Test 3: duplicate submission rejected ─────────────────────────────
        print("\n[Test 3] Duplicate submission rejection")
        pts = log_credit("client_A", 1, 5000, 30.0, tmp_db_path)
        assert pts == 0, "Duplicate should return 0 points"
        print("  ✓  Duplicate rejected correctly")

        # ── Test 4: log rounds ────────────────────────────────────────────────
        print("\n[Test 4] Logging round records")
        delta1 = log_round(1, started, 3, 0.714, tmp_db_path)
        delta2 = log_round(2, started, 3, 0.738, tmp_db_path)
        assert delta1 == 0.714,                   f"Round 1 delta should equal accuracy: {delta1}"
        assert abs(delta2 - 0.024) < 1e-5,        f"Round 2 delta wrong: {delta2}"
        print("  ✓  Round records inserted with correct deltas")

        # ── Test 5: leaderboard ───────────────────────────────────────────────
        print("\n[Test 5] Leaderboard query")
        board = get_leaderboard(tmp_db_path)
        assert len(board) == 3,                    "Expected 3 clients"
        assert board[0].total_points == 2000,      "Expected 2000 pts (2 rounds x 1000)"
        assert board[0].rounds_participated == 2,  "Expected 2 rounds"
        print(format_leaderboard(board, round_num=2))
        print("  ✓  Leaderboard correct")

        # ── Test 6: accuracy history ──────────────────────────────────────────
        print("\n[Test 6] Accuracy history")
        history = get_accuracy_history(tmp_db_path)
        assert len(history) == 2,          "Expected 2 rounds in history"
        assert history[0] == (1, 0.714),   f"Round 1 wrong: {history[0]}"
        print("  ✓  Accuracy history correct")

        # ── Test 7: submitted clients ─────────────────────────────────────────
        print("\n[Test 7] Submitted clients check")
        submitted = get_submitted_clients(1, tmp_db_path)
        assert set(submitted) == {"client_A", "client_B", "client_C"}
        print("  ✓  Submitted clients correct")

        # ── Test 8: client total credits ──────────────────────────────────────
        print("\n[Test 8] Client total credits")
        total = get_client_total_credits("client_A", tmp_db_path)
        assert total == 2000, f"Expected 2000, got {total}"
        print("  ✓  Client total credits correct")

        # ── Test 9: reset ─────────────────────────────────────────────────────
        print("\n[Test 9] Database reset")
        reset_db(tmp_db_path)
        board_after = get_leaderboard(tmp_db_path)
        assert len(board_after) == 0, "Leaderboard should be empty after reset"
        print("  ✓  Database reset correctly — leaderboard empty")

        # ── Test 10: final leaderboard format ─────────────────────────────────
        print("\n[Test 10] Final leaderboard formatting")
        # Repopulate for formatting test
        for client_id, samples in [("client_A", 5000), ("client_B", 4800), ("client_C", 5100)]:
            log_credit(client_id, 1, samples, 30.0, tmp_db_path)
        board_final = get_leaderboard(tmp_db_path)
        print(format_final_leaderboard(board_final))
        print("  ✓  Final leaderboard formatted correctly")

        # ── Test 11: credit formula ───────────────────────────────────────────
        print("\n[Test 11] Credit formula verification")
        assert compute_points(5000) == 1000
        assert compute_points(4800) == 960
        assert compute_points(5100) == 1020
        assert compute_points(0)    == 0
        print("  ✓  floor(samples / 5) verified")

    finally:
        os.unlink(tmp_db_path)

    print()
    print("=" * PRINT_WIDTH)
    print("  All checks passed. credits.py is ready.")
    print("=" * PRINT_WIDTH)