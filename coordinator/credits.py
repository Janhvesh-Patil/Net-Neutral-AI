import os
import sqlite3
import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# -- Database path -------------------------------------------------------------
COORDINATOR_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH         = os.path.join(COORDINATOR_DIR, "database.db")

# -- Credit formula ------------------------------------------------------------
def compute_points(samples_trained: int) -> int:
    return samples_trained // 5


# -- Result containers ---------------------------------------------------------

@dataclass
class ClientRecord:

    id:            int
    client_id:     str
    ip_address:    str
    registered_at: str
    last_seen:     str
    is_active:     bool


@dataclass
class CreditRecord:

    id:              int
    client_id:       str
    round:           int
    samples_trained: int
    time_seconds:    float
    points_earned:   int
    timestamp:       str


@dataclass
class RoundRecord:

    id:                int
    round_number:      int
    started_at:        str
    completed_at:      str
    clients_submitted: int
    global_accuracy:   float
    accuracy_delta:    float


@dataclass
class LeaderboardEntry:

    rank:               int
    client_id:          str
    total_points:       int
    total_samples:      int
    rounds_participated: int


# -----------------------------------------------------------------------------
# DATABASE CONNECTION HELPER
# -----------------------------------------------------------------------------

def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create a connection with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# -----------------------------------------------------------------------------
# DATABASE INITIALISATION
# -----------------------------------------------------------------------------

def init_db(db_path: str = DB_PATH) -> None:

    conn   = _connect(db_path)
    cursor = conn.cursor()

    # Enable WAL mode for better concurrent read performance
    cursor.execute("PRAGMA journal_mode=WAL")

    cursor.executescript("""
        -- 1. Clients table (NEW — replaces in-memory client_registry)
        CREATE TABLE IF NOT EXISTS clients (
            id            INTEGER  PRIMARY KEY AUTOINCREMENT,
            client_id     TEXT     NOT NULL UNIQUE,
            ip_address    TEXT     DEFAULT 'unknown',
            registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen     DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active     BOOLEAN  DEFAULT 1
        );

        -- 2. Rounds table (created before credits for FK reference)
        CREATE TABLE IF NOT EXISTS rounds (
            id                 INTEGER  PRIMARY KEY AUTOINCREMENT,
            round_number       INTEGER  NOT NULL UNIQUE,
            started_at         DATETIME NOT NULL,
            completed_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
            clients_submitted  INTEGER  NOT NULL DEFAULT 0,
            global_accuracy    REAL     NOT NULL DEFAULT 0.0,
            accuracy_delta     REAL     DEFAULT 0.0
        );

        -- 3. Credits table with FK constraints + UNIQUE duplicate guard
        CREATE TABLE IF NOT EXISTS credits (
            id               INTEGER  PRIMARY KEY AUTOINCREMENT,
            client_id        TEXT     NOT NULL,
            round            INTEGER  NOT NULL,
            samples_trained  INTEGER  NOT NULL DEFAULT 0,
            time_seconds     REAL     NOT NULL DEFAULT 0.0,
            points_earned    INTEGER  NOT NULL DEFAULT 0,
            timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(client_id, round),
            FOREIGN KEY (client_id) REFERENCES clients(client_id) ON DELETE CASCADE,
            FOREIGN KEY (round)     REFERENCES rounds(round_number) ON DELETE CASCADE
        );

        -- Indexes for fast queries
        CREATE INDEX IF NOT EXISTS idx_credits_client
            ON credits (client_id);

        CREATE INDEX IF NOT EXISTS idx_credits_round
            ON credits (round);

        CREATE INDEX IF NOT EXISTS idx_clients_active
            ON clients (is_active);
    """)

    conn.commit()
    conn.close()
    print(f"[Credits] [OK]  Database initialised at {db_path}")


def reset_db(db_path: str = DB_PATH) -> None:

    conn   = _connect(db_path)
    cursor = conn.cursor()
    # Drop in correct FK order (children first)
    cursor.executescript("""
        DROP TABLE IF EXISTS credits;
        DROP TABLE IF EXISTS rounds;
        DROP TABLE IF EXISTS clients;
    """)
    conn.commit()
    conn.close()
    init_db(db_path)
    print(f"[Credits] [OK]  Database reset complete")


# -----------------------------------------------------------------------------
# CLIENT OPERATIONS (NEW)
# -----------------------------------------------------------------------------

def register_client(
    client_id:  str,
    ip_address: str = "unknown",
    db_path:    str = DB_PATH,
) -> ClientRecord:
    """
    Register a client in the persistent clients table.
    If already exists, updates ip_address, last_seen, and reactivates.
    """
    conn   = _connect(db_path)
    cursor = conn.cursor()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
        INSERT INTO clients (client_id, ip_address, registered_at, last_seen, is_active)
        VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(client_id) DO UPDATE SET
            ip_address = excluded.ip_address,
            last_seen  = excluded.last_seen,
            is_active  = 1
        """,
        (client_id, ip_address, now, now)
    )
    conn.commit()

    # Fetch the full record to return
    cursor.execute(
        "SELECT id, client_id, ip_address, registered_at, last_seen, is_active "
        "FROM clients WHERE client_id = ?",
        (client_id,)
    )
    row = cursor.fetchone()
    conn.close()

    record = ClientRecord(
        id            = row[0],
        client_id     = row[1],
        ip_address    = row[2],
        registered_at = row[3],
        last_seen     = row[4],
        is_active     = bool(row[5]),
    )

    print(f"[Credits] [OK]  Client registered: {client_id} ({ip_address})")
    return record


def get_all_clients(
    active_only: bool = True,
    db_path:     str  = DB_PATH,
) -> List[ClientRecord]:
    """Return all registered clients from the database."""
    conn   = _connect(db_path)
    cursor = conn.cursor()

    if active_only:
        cursor.execute(
            "SELECT id, client_id, ip_address, registered_at, last_seen, is_active "
            "FROM clients WHERE is_active = 1 ORDER BY client_id"
        )
    else:
        cursor.execute(
            "SELECT id, client_id, ip_address, registered_at, last_seen, is_active "
            "FROM clients ORDER BY client_id"
        )

    rows = cursor.fetchall()
    conn.close()

    return [
        ClientRecord(
            id            = row[0],
            client_id     = row[1],
            ip_address    = row[2],
            registered_at = row[3],
            last_seen     = row[4],
            is_active     = bool(row[5]),
        )
        for row in rows
    ]


def update_client_last_seen(
    client_id: str,
    db_path:   str = DB_PATH,
) -> None:
    """Update the last_seen timestamp for a client."""
    conn   = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE clients SET last_seen = CURRENT_TIMESTAMP WHERE client_id = ?",
        (client_id,)
    )
    conn.commit()
    conn.close()


def deactivate_client(
    client_id: str,
    db_path:   str = DB_PATH,
) -> None:
    """Mark a client as inactive (soft delete)."""
    conn   = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE clients SET is_active = 0 WHERE client_id = ?",
        (client_id,)
    )
    conn.commit()
    conn.close()
    print(f"[Credits] [OK]  Client deactivated: {client_id}")


def get_client(
    client_id: str,
    db_path:   str = DB_PATH,
) -> Optional[ClientRecord]:
    """Fetch a single client by ID. Returns None if not found."""
    conn   = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, client_id, ip_address, registered_at, last_seen, is_active "
        "FROM clients WHERE client_id = ?",
        (client_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return ClientRecord(
        id            = row[0],
        client_id     = row[1],
        ip_address    = row[2],
        registered_at = row[3],
        last_seen     = row[4],
        is_active     = bool(row[5]),
    )


# -----------------------------------------------------------------------------
# AGGREGATE QUERIES
# -----------------------------------------------------------------------------

def get_client_total_credits(
    client_id: str,
    db_path:   str = DB_PATH,
) -> int:
    """Return the total lifetime points for a single client in this session."""
    conn   = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COALESCE(SUM(points_earned), 0) FROM credits WHERE client_id = ?",
        (client_id,)
    )
    total = cursor.fetchone()[0]
    conn.close()
    return total


def get_submitted_clients(
    round_num: int,
    db_path:   str = DB_PATH,
) -> set:
    """Return set of client_ids that have submitted for a given round."""
    conn   = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT client_id FROM credits WHERE round = ?",
        (round_num,)
    )
    result = {row[0] for row in cursor.fetchall()}
    conn.close()
    return result


# -----------------------------------------------------------------------------
# WRITE OPERATIONS
# -----------------------------------------------------------------------------

def ensure_round_exists(
    round_number:  int,
    started_at:    datetime.datetime,
    db_path:       str = DB_PATH,
) -> None:
    """
    FIX 1.3: Pre-create a round row if it doesn't already exist.

    This must be called before the first log_credit() for a given round,
    otherwise the FK constraint on credits(round) → rounds(round_number)
    will raise an IntegrityError.  The row is a placeholder — log_round()
    will UPDATE it later with the real accuracy and client count.
    """
    conn   = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO rounds
            (round_number, started_at, clients_submitted, global_accuracy, accuracy_delta)
        VALUES (?, ?, 0, 0.0, 0.0)
        """,
        (round_number, started_at.strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


def log_credit(
    client_id:       str,
    round_num:       int,
    samples_trained: int,
    time_seconds:    float,
    db_path:         str = DB_PATH,
) -> int:
    """
    Log a client's training contribution for a round.
    Uses UNIQUE(client_id, round) constraint — duplicates are silently ignored.
    Returns points earned (0 if duplicate).
    """
    points = compute_points(samples_trained)

    conn   = _connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO credits (client_id, round, samples_trained, time_seconds, points_earned)
            VALUES (?, ?, ?, ?, ?)
            """,
            (client_id, round_num, samples_trained, round(time_seconds, 2), points)
        )
        conn.commit()
        conn.close()

        print(f"[Credits] [OK]  Logged: {client_id} | round {round_num} | "
              f"{samples_trained} samples | {points} pts")
        return points

    except sqlite3.IntegrityError as e:
        conn.close()
        if "UNIQUE constraint" in str(e):
            print(f"[Credits] [!]  Duplicate submission ignored: {client_id} round {round_num}")
            return 0
        else:
            # FK violation or other integrity error
            print(f"[Credits] [!]  Integrity error: {e}")
            raise


def log_round(
    round_number:      int,
    started_at:        datetime.datetime,
    clients_submitted: int,
    global_accuracy:   float,
    db_path:           str = DB_PATH,
) -> float:

    prev_accuracy = _get_previous_accuracy(round_number, db_path)
    delta         = global_accuracy - prev_accuracy

    conn   = _connect(db_path)
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
    print(f"[Credits] [OK]  Round {round_number} logged | "
          f"accuracy: {global_accuracy*100:.2f}% ({delta_str}) | "
          f"{clients_submitted} clients")
    return delta


# -----------------------------------------------------------------------------
# READ OPERATIONS
# -----------------------------------------------------------------------------

def get_leaderboard(db_path: str = DB_PATH) -> List[LeaderboardEntry]:

    conn   = _connect(db_path)
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

    conn   = _connect(db_path)
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

    conn   = _connect(db_path)
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

    conn   = _connect(db_path)
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

    conn   = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COALESCE(SUM(points_earned), 0) FROM credits WHERE client_id = ?",
        (client_id,)
    )
    total = cursor.fetchone()[0]
    conn.close()
    return total


# -----------------------------------------------------------------------------
# FORMATTING HELPERS
# -----------------------------------------------------------------------------

PRINT_WIDTH = 60

def _line(char="-"): return char * PRINT_WIDTH

def format_leaderboard(entries: List[LeaderboardEntry], round_num: int) -> str:
    lines = []
    lines.append(_line("-"))
    lines.append(f"  Leaderboard after Round {round_num}")
    lines.append(_line("-"))
    lines.append(f"  {'Rank':<6}{'Client':<14}{'Points':>10}{'Samples':>12}{'Rounds':>8}")
    lines.append(_line("-"))
    for e in entries:
        lines.append(
            f"  {e.rank:<6}{e.client_id:<14}"
            f"{e.total_points:>10,}"
            f"{e.total_samples:>12,}"
            f"{e.rounds_participated:>8}"
        )
    lines.append(_line("-"))
    return "\n".join(lines)


def format_final_leaderboard(entries: List[LeaderboardEntry]) -> str:

    lines = []
    lines.append(_line("="))
    lines.append("  FINAL LEADERBOARD")
    lines.append(_line("="))
    lines.append(
        f"  {'Rank':<6}{'Client':<14}{'Total Pts':>12}"
        f"{'Total Samples':>15}{'Rounds':>8}"
    )
    lines.append(_line("-"))
    for e in entries:
        medal = ["Gold", "Silver", "Bronze"][e.rank - 1] if e.rank <= 3 else "  "
        lines.append(
            f"  {medal} {e.rank:<4}{e.client_id:<14}"
            f"{e.total_points:>12,}"
            f"{e.total_samples:>15,}"
            f"{e.rounds_participated:>8}"
        )
    lines.append(_line("="))
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# PRIVATE HELPERS
# -----------------------------------------------------------------------------

def _get_previous_accuracy(round_number: int, db_path: str) -> float:
    if round_number <= 1:
        return 0.0
    conn   = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT global_accuracy FROM rounds WHERE round_number = ?",
        (round_number - 1,)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0.0


# -----------------------------------------------------------------------------
# SANITY CHECK
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    PRINT_WIDTH = 60
    print("=" * PRINT_WIDTH)
    print("  credits.py sanity check (improved schema)")
    print("=" * PRINT_WIDTH)

    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db_path = tmp_db.name
    tmp_db.close()

    try:
        # -- Test 1: init ------------------------------------------------------
        print("\n[Test 1] Database initialisation")
        init_db(tmp_db_path)

        # Verify tables exist
        conn = _connect(tmp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        assert "clients" in tables, f"clients table missing, got: {tables}"
        assert "credits" in tables, f"credits table missing, got: {tables}"
        assert "rounds" in tables, f"rounds table missing, got: {tables}"

        # Verify FK is enabled
        cursor.execute("PRAGMA foreign_keys")
        fk_status = cursor.fetchone()[0]
        assert fk_status == 1, f"Foreign keys not enabled: {fk_status}"
        conn.close()
        print("  [OK]  Tables created: clients, credits, rounds")
        print("  [OK]  Foreign keys enabled")

        # -- Test 2: register clients ------------------------------------------
        print("\n[Test 2] Client registration")
        rec_a = register_client("client_A", "10.0.0.100", tmp_db_path)
        rec_b = register_client("client_B", "10.0.0.101", tmp_db_path)
        rec_c = register_client("client_C", "10.0.0.102", tmp_db_path)
        assert rec_a.client_id == "client_A"
        assert rec_a.ip_address == "10.0.0.100"
        assert rec_a.is_active is True
        print("  [OK]  3 clients registered with IPs")

        # Test re-registration (update IP)
        rec_a2 = register_client("client_A", "10.0.0.200", tmp_db_path)
        assert rec_a2.ip_address == "10.0.0.200", f"IP not updated: {rec_a2.ip_address}"
        print("  [OK]  Re-registration updates IP correctly")

        # -- Test 3: get all clients -------------------------------------------
        print("\n[Test 3] Get all clients")
        clients = get_all_clients(db_path=tmp_db_path)
        assert len(clients) == 3, f"Expected 3 clients, got {len(clients)}"
        print(f"  [OK]  {len(clients)} active clients returned")

        # -- Test 4: deactivate client -----------------------------------------
        print("\n[Test 4] Deactivate client")
        deactivate_client("client_C", tmp_db_path)
        active = get_all_clients(active_only=True, db_path=tmp_db_path)
        all_clients = get_all_clients(active_only=False, db_path=tmp_db_path)
        assert len(active) == 2, f"Expected 2 active, got {len(active)}"
        assert len(all_clients) == 3, f"Expected 3 total, got {len(all_clients)}"
        # Reactivate for remaining tests
        register_client("client_C", "10.0.0.102", tmp_db_path)
        print("  [OK]  Deactivation/reactivation works correctly")

        # -- Test 5: log rounds (must exist before credits due to FK) ----------
        print("\n[Test 5] Logging rounds")
        started = datetime.datetime(2026, 6, 1, 10, 0, 0)
        delta1 = log_round(1, started, 3, 0.714, tmp_db_path)
        delta2 = log_round(2, started, 3, 0.738, tmp_db_path)
        assert delta1 == 0.714, f"Round 1 delta should equal accuracy: {delta1}"
        assert abs(delta2 - 0.024) < 1e-5, f"Round 2 delta wrong: {delta2}"
        print("  [OK]  Round records inserted with correct deltas")

        # -- Test 6: log credits -----------------------------------------------
        print("\n[Test 6] Logging credits for 3 clients, 2 rounds")
        for round_num in [1, 2]:
            for client_id, samples in [("client_A", 5000), ("client_B", 5000), ("client_C", 5000)]:
                pts = log_credit(client_id, round_num, samples, 30.0, tmp_db_path)
                assert pts == 1000, f"Expected 1000 pts, got {pts}"
        print("  [OK]  All credit records inserted correctly")

        # -- Test 7: UNIQUE constraint (duplicate rejection at DB level) -------
        print("\n[Test 7] UNIQUE constraint — duplicate rejection")
        pts = log_credit("client_A", 1, 5000, 30.0, tmp_db_path)
        assert pts == 0, "Duplicate should return 0 points"
        print("  [OK]  Duplicate rejected by UNIQUE(client_id, round) constraint")

        # -- Test 8: FK constraint enforcement ---------------------------------
        print("\n[Test 8] Foreign key constraint enforcement")
        try:
            # Try to insert credit for non-existent client
            conn = _connect(tmp_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO credits (client_id, round, samples_trained, points_earned) "
                "VALUES ('fake_client', 1, 100, 20)"
            )
            conn.commit()
            conn.close()
            assert False, "FK violation should have raised an error"
        except sqlite3.IntegrityError:
            conn.close()
            print("  [OK]  FK violation correctly rejected (non-existent client)")

        try:
            # Try to insert credit for non-existent round
            conn = _connect(tmp_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO credits (client_id, round, samples_trained, points_earned) "
                "VALUES ('client_A', 99, 100, 20)"
            )
            conn.commit()
            conn.close()
            assert False, "FK violation should have raised an error"
        except sqlite3.IntegrityError:
            conn.close()
            print("  [OK]  FK violation correctly rejected (non-existent round)")

        # -- Test 9: leaderboard -----------------------------------------------
        print("\n[Test 9] Leaderboard query")
        board = get_leaderboard(tmp_db_path)
        assert len(board) == 3, "Expected 3 clients"
        assert board[0].total_points == 2000, "Expected 2000 pts (2 rounds x 1000)"
        assert board[0].rounds_participated == 2, "Expected 2 rounds"
        print(format_leaderboard(board, round_num=2))
        print("  [OK]  Leaderboard correct")

        # -- Test 10: accuracy history -----------------------------------------
        print("\n[Test 10] Accuracy history")
        history = get_accuracy_history(tmp_db_path)
        assert len(history) == 2, "Expected 2 rounds in history"
        assert history[0] == (1, 0.714), f"Round 1 wrong: {history[0]}"
        print("  [OK]  Accuracy history correct")

        # -- Test 11: get_client single lookup ---------------------------------
        print("\n[Test 11] Single client lookup")
        c = get_client("client_B", tmp_db_path)
        assert c is not None
        assert c.client_id == "client_B"
        assert c.ip_address == "10.0.0.101"
        missing = get_client("nonexistent", tmp_db_path)
        assert missing is None
        print("  [OK]  Single client lookup correct")

        # -- Test 12: credit formula -------------------------------------------
        print("\n[Test 12] Credit formula verification")
        assert compute_points(5000) == 1000
        assert compute_points(4800) == 960
        assert compute_points(5100) == 1020
        assert compute_points(0)    == 0
        print("  [OK]  floor(samples / 5) verified")

        # -- Test 13: reset ----------------------------------------------------
        print("\n[Test 13] Database reset")
        reset_db(tmp_db_path)
        board_after = get_leaderboard(tmp_db_path)
        assert len(board_after) == 0, "Leaderboard should be empty after reset"
        clients_after = get_all_clients(active_only=False, db_path=tmp_db_path)
        assert len(clients_after) == 0, "Clients should be empty after reset"
        print("  [OK]  Database reset correctly — all tables empty")

        # -- Test 14: CASCADE delete -------------------------------------------
        print("\n[Test 14] CASCADE delete verification")
        # Re-register, add round, add credit, delete client
        register_client("cascade_test", "1.2.3.4", tmp_db_path)
        log_round(1, started, 1, 0.5, tmp_db_path)
        log_credit("cascade_test", 1, 1000, 10.0, tmp_db_path)

        # Verify credit exists
        total = get_client_total_credits("cascade_test", tmp_db_path)
        assert total == 200, f"Expected 200, got {total}"

        # Delete the client → credits should cascade
        conn = _connect(tmp_db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM clients WHERE client_id = 'cascade_test'")
        conn.commit()

        # Verify credit was cascade-deleted
        cursor.execute("SELECT COUNT(*) FROM credits WHERE client_id = 'cascade_test'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0, f"Expected 0 credits after cascade, got {count}"
        print("  [OK]  CASCADE delete works — client deletion removes credits")

    finally:
        os.unlink(tmp_db_path)

    print()
    print("=" * PRINT_WIDTH)
    print("  All checks passed. credits.py (improved schema) is ready.")
    print("=" * PRINT_WIDTH)