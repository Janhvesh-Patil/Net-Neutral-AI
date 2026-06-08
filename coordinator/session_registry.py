"""
session_registry.py — Multi-coordinator session discovery.

Each coordinator registers an active session (Supabase when configured).
Clients list sessions via GET /api/lobby (cloud registry + optional LAN scan).
"""

from __future__ import annotations

import datetime
import os
import sys
import uuid
from typing import Dict, List, Optional

# Allow importing shared config from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import config  # noqa: E402
from shared import ip_utils  # noqa: E402

TABLE_NAME = "active_sessions"

# Set once per coordinator process
_session_id: Optional[str] = None
_session_name: Optional[str] = None


def get_session_id() -> str:
    global _session_id
    if _session_id is None:
        _session_id = str(uuid.uuid4())
    return _session_id


def set_session_name(name: str) -> None:
    global _session_name
    _session_name = name.strip() or None


def get_session_name() -> str:
    if _session_name:
        return _session_name
    try:
        hostname = ip_utils.get_local_hostname()
    except Exception:
        hostname = "Coordinator"
    return f"{hostname} — Net-Neutral Session"


def _get_supabase_client():
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    except ImportError:
        print("[SessionRegistry] [!] supabase-py not installed")
        return None
    except Exception as exc:
        print(f"[SessionRegistry] [!] Supabase client error: {exc}")
        return None


def build_session_payload(
    *,
    round_status: str,
    current_round: int,
    connected_clients: int,
    max_clients: int,
    coordinator_ip: Optional[str] = None,
    port: Optional[int] = None,
) -> Dict:
    ip = coordinator_ip or ip_utils.get_local_ip()
    port = port or config.COORDINATOR_PORT
    base_url = f"http://{ip}:{port}"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {
        "session_id": get_session_id(),
        "session_name": get_session_name(),
        "coordinator_ip": ip,
        "coordinator_port": port,
        "base_url": base_url,
        "host_name": ip_utils.get_local_hostname(),
        "round_status": round_status,
        "current_round": current_round,
        "connected_clients": connected_clients,
        "max_clients": max_clients,
        "last_heartbeat": now,
    }


def register_session(payload: Dict) -> bool:
    supabase = _get_supabase_client()
    if supabase is None:
        return False

    try:
        row = {**payload, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        supabase.table(TABLE_NAME).upsert(row, on_conflict="session_id").execute()
        print(f"[SessionRegistry] [OK] Registered session {payload['session_id'][:8]}…")
        return True
    except Exception as exc:
        print(f"[SessionRegistry] [!] Register failed: {exc}")
        return False


def heartbeat_session(payload: Dict) -> bool:
    supabase = _get_supabase_client()
    if supabase is None:
        return False

    try:
        supabase.table(TABLE_NAME).upsert(payload, on_conflict="session_id").execute()
        return True
    except Exception as exc:
        print(f"[SessionRegistry] [!] Heartbeat failed: {exc}")
        return False


def deregister_session(session_id: Optional[str] = None) -> bool:
    sid = session_id or get_session_id()
    supabase = _get_supabase_client()
    if supabase is None:
        return False

    try:
        supabase.table(TABLE_NAME).delete().eq("session_id", sid).execute()
        print(f"[SessionRegistry] [OK] Deregistered session {sid[:8]}…")
        return True
    except Exception as exc:
        print(f"[SessionRegistry] [!] Deregister failed: {exc}")
        return False


def list_cloud_sessions(stale_after_secs: Optional[int] = None) -> List[Dict]:
    stale_after_secs = stale_after_secs or config.SESSION_STALE_SECS
    supabase = _get_supabase_client()
    if supabase is None:
        return []

    try:
        response = (
            supabase.table(TABLE_NAME)
            .select("*")
            .order("last_heartbeat", desc=True)
            .execute()
        )
        rows = response.data or []
    except Exception as exc:
        print(f"[SessionRegistry] [!] List failed: {exc}")
        return []

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=stale_after_secs)
    active: List[Dict] = []

    for row in rows:
        heartbeat = row.get("last_heartbeat")
        if not heartbeat:
            continue
        try:
            ts = datetime.datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.timezone.utc)
            if ts >= cutoff:
                active.append(row)
        except ValueError:
            active.append(row)

    return active


def merge_session_lists(*lists: List[Dict]) -> List[Dict]:
    """Deduplicate sessions by session_id, preferring newer heartbeats."""
    merged: Dict[str, Dict] = {}
    for session_list in lists:
        for session in session_list:
            sid = session.get("session_id")
            if not sid:
                continue
            existing = merged.get(sid)
            if existing is None:
                merged[sid] = session
                continue
            if (session.get("last_heartbeat") or "") >= (existing.get("last_heartbeat") or ""):
                merged[sid] = session

    result = list(merged.values())
    result.sort(key=lambda s: s.get("session_name") or "")
    return result
