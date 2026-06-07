"""
supabase_sync.py — Syncs local session credits to a central Supabase database.

Called automatically when a federated training session completes (round_status == 'done').
If Supabase credentials are not configured, fails gracefully with a warning.

Environment variables:
    SUPABASE_URL  — Your Supabase project URL (e.g. https://xyz.supabase.co)
    SUPABASE_KEY  — Your Supabase anon/public key

Supabase table schema (run in SQL Editor):
    CREATE TABLE global_credits (
        id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
        client_id       TEXT        NOT NULL UNIQUE,
        total_points    BIGINT      NOT NULL DEFAULT 0,
        total_samples   BIGINT      NOT NULL DEFAULT 0,
        total_rounds    INTEGER     NOT NULL DEFAULT 0,
        last_session_at TIMESTAMPTZ DEFAULT NOW(),
        created_at      TIMESTAMPTZ DEFAULT NOW()
    );

    ALTER TABLE global_credits ENABLE ROW LEVEL SECURITY;

    CREATE POLICY "Allow anonymous access" ON global_credits
        FOR ALL USING (true) WITH CHECK (true);
"""

import os
import datetime
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


def _get_supabase_client():
    """
    Lazily initialise and return a Supabase client.
    Returns None if credentials are not configured.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None

    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except ImportError:
        print("[SupabaseSync] [!] supabase-py not installed. Run: pip install supabase")
        return None
    except Exception as e:
        print(f"[SupabaseSync] [!] Failed to create Supabase client: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SYNC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def sync_credits_to_cloud(db_path: Optional[str] = None) -> bool:
    """
    Sync local session credits to the central Supabase global_credits table.

    For each client in the local leaderboard:
    - If the client exists in Supabase: INCREMENT total_points, total_samples, total_rounds
    - If the client is new: INSERT a new row

    Args:
        db_path: Path to local SQLite database (default: credits.DB_PATH)

    Returns:
        True if sync succeeded, False otherwise
    """
    import credits

    if db_path is None:
        db_path = credits.DB_PATH

    # Check Supabase credentials
    supabase = _get_supabase_client()
    if supabase is None:
        print("[SupabaseSync] [!] Supabase not configured. "
              "Set SUPABASE_URL and SUPABASE_KEY environment variables.")
        print("[SupabaseSync] [!] Credits saved locally only. Cloud sync skipped.")
        return False

    # Read local leaderboard
    leaderboard = credits.get_leaderboard(db_path)
    if not leaderboard:
        print("[SupabaseSync] [!] No credits to sync (leaderboard empty)")
        return False

    print(f"\n[SupabaseSync] Syncing {len(leaderboard)} client(s) to Supabase...")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    success_count = 0

    for entry in leaderboard:
        try:
            # Check if client already exists in global_credits
            response = (
                supabase.table("global_credits")
                .select("client_id, total_points, total_samples, total_rounds")
                .eq("client_id", entry.client_id)
                .execute()
            )

            if response.data and len(response.data) > 0:
                # Client exists — INCREMENT values
                existing = response.data[0]
                new_points  = existing["total_points"] + entry.total_points
                new_samples = existing["total_samples"] + entry.total_samples
                new_rounds  = existing["total_rounds"] + entry.rounds_participated

                supabase.table("global_credits").update({
                    "total_points":    new_points,
                    "total_samples":   new_samples,
                    "total_rounds":    new_rounds,
                    "last_session_at": now,
                }).eq("client_id", entry.client_id).execute()

                print(f"  [OK] {entry.client_id}: updated "
                      f"(+{entry.total_points} pts, total: {new_points})")
            else:
                # New client — INSERT
                supabase.table("global_credits").insert({
                    "client_id":       entry.client_id,
                    "total_points":    entry.total_points,
                    "total_samples":   entry.total_samples,
                    "total_rounds":    entry.rounds_participated,
                    "last_session_at": now,
                }).execute()

                print(f"  [OK] {entry.client_id}: created "
                      f"({entry.total_points} pts)")

            success_count += 1

        except Exception as e:
            print(f"  [!] {entry.client_id}: sync failed — {e}")

    print(f"[SupabaseSync] [OK] Sync complete: "
          f"{success_count}/{len(leaderboard)} clients synced")

    return success_count == len(leaderboard)


# ─────────────────────────────────────────────────────────────────────────────
# CLOUD LEADERBOARD (READ)
# ─────────────────────────────────────────────────────────────────────────────

def get_global_leaderboard(limit: int = 50) -> list:
    """
    Fetch the global lifetime leaderboard from Supabase.

    Returns:
        List of dicts: [{client_id, total_points, total_samples, total_rounds, last_session_at}, ...]
        Empty list if Supabase not configured or query fails.
    """
    supabase = _get_supabase_client()
    if supabase is None:
        return []

    try:
        response = (
            supabase.table("global_credits")
            .select("client_id, total_points, total_samples, total_rounds, last_session_at")
            .order("total_points", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []

    except Exception as e:
        print(f"[SupabaseSync] [!] Failed to fetch global leaderboard: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  supabase_sync.py — configuration check")
    print("=" * 60)

    print(f"\n  SUPABASE_URL: {'[SET]' if SUPABASE_URL else '[NOT SET]'}")
    print(f"  SUPABASE_KEY: {'[SET]' if SUPABASE_KEY else '[NOT SET]'}")

    if SUPABASE_URL and SUPABASE_KEY:
        print("\n  Attempting connection...")
        client = _get_supabase_client()
        if client:
            print("  [OK] Supabase client created successfully")

            # Try fetching global leaderboard
            board = get_global_leaderboard(limit=5)
            print(f"  [OK] Global leaderboard: {len(board)} entries")
            for entry in board:
                print(f"    {entry['client_id']}: {entry['total_points']} pts")
        else:
            print("  [!] Failed to create client")
    else:
        print("\n  To configure, set environment variables:")
        print("    set SUPABASE_URL=https://your-project.supabase.co")
        print("    set SUPABASE_KEY=your-anon-key")
        print("\n  Or in PowerShell:")
        print("    $env:SUPABASE_URL='https://your-project.supabase.co'")
        print("    $env:SUPABASE_KEY='your-anon-key'")

    print("\n" + "=" * 60)
    print("  Check complete.")
    print("=" * 60)
