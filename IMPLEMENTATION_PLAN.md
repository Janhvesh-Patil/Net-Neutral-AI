# Net-Neutral AI — Implementation Plan

**Branch:** `scope_creep`  
**Last updated:** June 7, 2026  
**Status:** Active roadmap

---

## Overview

This document tracks planned enhancements for Net-Neutral AI after the `scope_creep` automation release. Work is grouped by priority. **Phase 1** fixes correctness bugs that block reliable multi-round federated training. **Phase 2** adds multi-coordinator session discovery (implemented in this sprint). **Phase 3+** covers UX, scale, and production hardening.

---

## Phase 1 — Critical Bug Fixes (P0) ✅ Complete

These must land before demo recordings or merging to `main`.

| # | Issue | Impact | Proposed fix | Files |
|---|-------|--------|--------------|-------|
| 1.1 | **Round 2+ aggregation blocked** — `check_round_completion()` returns early when `round_status == 'waiting_for_clients'`, so FedAvg never runs after round 1 | Training appears to run but global model never updates past round 1 | After round 1, transition to `active` (skip re-distribution). Only use `data_distributing` on first `/start_training` | `backend/coordinator/server.py` |
| 1.2 | **Shard dataloader mismatch** — `get_client_dataloader()` slices by hardcoded `SHARD_RANGES` even when a distributed CSV shard is loaded | `client_B` / `client_C` may train on empty data | If `local_shard_path` is set, use `get_full_dataloader()` instead of index slicing | `backend/client/client.py`, `backend/client/data.py` |
| 1.3 | **Credit FK ordering** — `log_credit()` on `/submit` runs before `log_round()` creates the round row | SQLite FK errors on first submission per round | Call `ensure_round_exists(round_num)` at round start or before first `log_credit` | `backend/coordinator/credits.py`, `backend/coordinator/server.py` |
| 1.4 | **Data shard race** — clients download shard immediately after register, before coordinator clicks Start Training | Client crashes on first run | Poll `/status` until `data_distributing` or `active`, honour `WAIT_FOR_DATA_TIMEOUT_SECS` | `backend/client/client.py` |
| 1.5 | **Incomplete coordinator deps** — `pandas`, `scikit-learn` missing from `requirements.txt` | Data upload / sharding fails on fresh install | Add to `coordinator/requirements.txt` | `coordinator/requirements.txt` |

**Acceptance criteria (Phase 1):**
- [x] Full 5-round session completes with improving or stable global accuracy
- [x] All 3 clients train on non-empty shards from uploaded CSV
- [x] Credits logged for every round without FK errors
- [x] Fresh `pip install -r requirements.txt` supports dataset upload

**Estimated effort:** 1–2 days  
**Status:** ✅ Complete (June 8, 2026)

---

## Phase 2 — Multi-Coordinator Session Lobby (P1) ✅ In progress

### Problem

A WiFi network may host multiple independent training sessions (multiple coordinators). Clients should not need to know a coordinator's IP in advance — they should browse available sessions and join one.

### Solution architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Session discovery                        │
├──────────────────────────┬──────────────────────────────────┤
│  Supabase registry       │  LAN subnet scan (fallback)       │
│  active_sessions table   │  GET /api/session_info per host   │
│  heartbeat every 10s     │  triggered via GET /api/lobby     │
└──────────────────────────┴──────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Client lobby UI  │
                    │  list + Join btn  │
                    └──────────────────┘
```

### Backend (implemented)

| Component | Purpose |
|-----------|---------|
| `backend/coordinator/session_registry.py` | Register session, heartbeat, list/deregister via Supabase |
| `backend/coordinator/lan_scan.py` | Probe subnet for coordinators (`/api/session_info`) |
| `GET /api/session_info` | This coordinator's live session metadata |
| `GET /api/lobby` | Merged session list (Supabase + optional `?subnet=192.168.1`) |
| `GET /api/public_config` | Discovery settings for frontend |
| Heartbeat thread | Updates registry every 10s while server runs |

**Supabase table** (`active_sessions`):

```sql
CREATE TABLE active_sessions (
    session_id        TEXT        PRIMARY KEY,
    session_name      TEXT        NOT NULL,
    coordinator_ip    TEXT        NOT NULL,
    coordinator_port  INTEGER     NOT NULL DEFAULT 5000,
    base_url          TEXT        NOT NULL,
    host_name         TEXT,
    round_status      TEXT        DEFAULT 'waiting_for_clients',
    current_round     INTEGER     DEFAULT 1,
    connected_clients INTEGER     DEFAULT 0,
    max_clients       INTEGER     DEFAULT 3,
    last_heartbeat    TIMESTAMPTZ DEFAULT NOW(),
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE active_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow anonymous access" ON active_sessions
    FOR ALL USING (true) WITH CHECK (true);
```

### Frontend (implemented)

| Screen | Behaviour |
|--------|-----------|
| Client → **Session Lobby** | Lists active sessions as cards (name, host, clients, status) |
| Refresh | Fetches `GET /api/lobby` via discovery host + optional subnet scan |
| Join | Sets `coordinatorURL`, registers client, opens client dashboard |
| Manual connect | Fallback link for direct IP entry |

### CLI client (implemented)

```bash
# List sessions on the LAN
python client.py --list_sessions --discovery_host 192.168.1.10:5000 --subnet 192.168.1

# Join a specific coordinator
python client.py --client_id client_A --coordinator_url http://192.168.1.10:5000
```

### Acceptance criteria (Phase 2)

- [ ] Two coordinators on same subnet both appear in lobby after refresh
- [ ] Client can join either session; credits tracked per coordinator DB
- [ ] Stale sessions disappear from lobby ~45s after coordinator stops
- [ ] Works without Supabase when subnet scan is used
- [ ] Works with Supabase when `SUPABASE_URL` / `SUPABASE_KEY` are set

**Estimated effort:** 2–3 days (core done; polish + tests remain)

---

## Phase 3 — UX & Observability (P2)

| # | Feature | Description | Effort |
|---|---------|-------------|--------|
| 3.1 | **WebSocket live updates** | Replace 3–5s polling with SSE or WebSocket for round status | 2d |
| 3.2 | **Accuracy chart** | Line chart of global accuracy per round on coordinator dashboard | 1d |
| 3.3 | **Global leaderboard UI** | Show Supabase `global_credits` in coordinator + client dashboards | 1d |
| 3.4 | **Browser client training bridge** | Clear UX: web client = monitor only; link to run `client.py` with copied args | 0.5d |
| 3.5 | **Session naming** | Let coordinator set a custom session name in dashboard | 0.5d |
| 3.6 | **Results export** | Download rounds/credits as CSV from dashboard | 1d |

---

## Phase 4 — Scale & Reliability (P2)

| # | Feature | Description | Effort |
|---|---------|-------------|--------|
| 4.1 | **>3 clients** | Raise `MAX_CLIENTS`, dynamic shard split, UI client count | 2d |
| 4.2 | **Async / partial rounds** | FedAvg when K of N clients submit (timeout-based) | 2d |
| 4.3 | **Coordinator health** | Heartbeat timeout, client reconnect, stale weight cleanup | 1d |
| 4.4 | **CI for scope_creep** | Extend lint workflow to `scope_creep` branch | 0.5d |
| 4.5 | **End-to-end test** | pytest harness: mock 3 clients, 2 rounds, assert accuracy logged | 2d |

---

## Phase 5 — Privacy & Production (P3)

| # | Feature | Description | Effort |
|---|---------|-------------|--------|
| 5.1 | **Gradient clipping / DP noise** | Optional differential privacy on submitted weights | 3d |
| 5.2 | **TLS / HTTPS** | Reverse proxy or Flask TLS for non-LAN deploys | 1d |
| 5.3 | **Auth & session tokens** | Coordinator password; client join tokens | 3d |
| 5.4 | **mDNS discovery** | `_net-neutral._tcp.local` for zero-config LAN discovery | 2d |
| 5.5 | **Public installer** | PyInstaller / MSI for one-click client node | 3d |

---

## Recommended execution order

```
Phase 1 (bugs)  →  Phase 2 (lobby)  →  Phase 3.1–3.3 (UX)
        ↓
Phase 4.1 (scale)  →  merge to main  →  Phase 5 as roadmap
```

---

## Configuration reference (new)

| Setting | Location | Default | Purpose |
|---------|----------|---------|---------|
| `SESSION_HEARTBEAT_SECS` | `config.py` | `10` | Registry heartbeat interval |
| `SESSION_STALE_SECS` | `config.py` | `45` | Hide sessions older than this |
| `DISCOVERY_PORT` | `config.py` | `5000` | Port for LAN scan |
| `SUPABASE_URL` / `SUPABASE_KEY` | env | empty | Cloud registry + credits |
| `--coordinator_url` | `client.py` CLI | from config | Override coordinator target |
| `--list_sessions` | `client.py` CLI | off | Print lobby and exit |
| `--subnet` | `client.py` CLI | auto | Subnet prefix for LAN scan |

---

## Testing checklist (before merge)

- [ ] `python integration_test.py` — 6/6 pass
- [ ] `python backend/coordinator/credits.py` — sanity tests pass
- [ ] Two coordinators + lobby refresh shows both
- [ ] Client joins correct session; training completes 5 rounds (after Phase 1 fixes)
- [ ] Supabase sync still runs on session complete
- [ ] Manual IP fallback still works

---

*Net-Neutral AI — GitHub DevDays Hackathon 2026*
