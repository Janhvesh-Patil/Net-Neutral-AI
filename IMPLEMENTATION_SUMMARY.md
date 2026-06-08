# Net-Neutral AI: Implementation Summary

**Date**: June 7, 2026  
**Status**: All Tasks Completed  
**Branch**: `scope_creep`

---

## Architecture

```
                    +---------------------------+
                    |   Coordinator (Flask)     |
                    |   coordinator/server.py   |
                    |                           |
                    |  +--------+  +---------+  |         +------------------+
                    |  | FedAvg |  | Credits |  | ------> | Supabase Cloud   |
                    |  | Engine |  | SQLite  |  |  sync   | global_credits   |
                    |  +--------+  +---------+  |         +------------------+
                    +------------+--------------+
                                 | WiFi / LAN
              +------------------+-------------------+
              |                  |                   |
    +---------v------+  +--------v-------+  +--------v-------+
    |   Client A     |  |   Client B     |  |   Client C     |
    |   client.py    |  |   client.py    |  |   client.py    |
    |   Local data   |  |   Local data   |  |   Local data   |
    +----------------+  +----------------+  +----------------+
```

## Databases

### Local SQLite (`coordinator/database.db`)
- **clients** table: Registered clients with IP, status
- **rounds** table: Round metadata with accuracy tracking
- **credits** table: Per-client per-round contribution credits
- Foreign keys enforced, UNIQUE constraints, CASCADE deletes

### Cloud Supabase (`global_credits`)
- Accumulates lifetime credits across all training sessions
- Auto-synced when a training session completes
- Falls back gracefully when not configured

---

## Frontend Flow

```
[Landing Page]  →  Choose Role
    |
    ├── Coordinator
    |     ├── Enter IP + Port + Client Count (1-3)
    |     └── Dashboard: Upload data, monitor clients, start training
    |
    └── Client
          ├── Enter Coordinator IP + Port + Client Name (A/B/C)
          └── Dashboard: View status, credits, round progress
```

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/register` | Register a client (ID + IP) |
| GET | `/model` | Download global model weights |
| POST | `/submit` | Submit trained weights |
| GET | `/status` | Poll training status |
| GET | `/results` | Final accuracy + round count |
| GET | `/api/config` | Server config (max_clients, total_rounds) |
| GET | `/api/client_status/<id>` | Individual client credits + status |
| POST | `/upload_dataset` | Upload CSV dataset |
| GET | `/get_clients` | List registered clients |
| POST | `/start_training` | Begin data distribution + training |
| POST | `/get_data_shard` | Download client's data shard |
| GET | `/` | Serve frontend HTML |

---

## Key Files

| File | Purpose |
|------|---------|
| `coordinator/server.py` | Flask server, API endpoints, state machine |
| `coordinator/credits.py` | SQLite database operations, leaderboard |
| `coordinator/supabase_sync.py` | Cloud credit sync to Supabase |
| `coordinator/fedavg.py` | Federated averaging algorithm |
| `coordinator/evaluate.py` | Model evaluation |
| `coordinator/data_distributor.py` | Dataset splitting for clients |
| `client/client.py` | Client training loop |
| `client/model.py` | Transformer classifier model |
| `client/train.py` | Local training logic |
| `client/data.py` | Data loading (supports CSV shards) |
| `shared/config.py` | All configuration parameters |
| `shared/ip_utils.py` | IP address auto-discovery |
| `frontend/index.html` | Web UI with role selection |
| `frontend/styles.css` | Styling (purple gradient theme) |
| `frontend/app.js` | Frontend logic + API integration |

---

## Quick Start

```bash
# 1. Start coordinator
python coordinator/server.py

# 2. Open browser
#    http://localhost:5000/
#    → Select "Coordinator"
#    → Enter IP, select client count, launch dashboard
#    → Upload dataset CSV

# 3. Start clients (in separate terminals)
python client/client.py --client_id client_A
python client/client.py --client_id client_B
python client/client.py --client_id client_C

# 4. Click "Start Training" in the dashboard
```

---

## Dependencies

```
flask==3.0.3
torch==2.3.0
numpy==1.26.4
supabase>=2.0.0
pandas>=1.3.0
scikit-learn>=0.24.0
```

---

## Phase 1 Bug Fixes (June 8, 2026)

Five critical correctness bugs have been fixed:

| # | Bug | Fix | Files |
|---|-----|-----|-------|
| 1.1 | Round 2+ aggregation blocked | Transition to `active` after round 1 (not `waiting_for_clients`) | `coordinator/server.py` |
| 1.2 | Shard dataloader mismatch | Use `get_full_dataloader()` when `local_shard_path` is set | `client/client.py` |
| 1.3 | Credit FK ordering | New `ensure_round_exists()` pre-creates round row before credits | `coordinator/credits.py`, `coordinator/server.py` |
| 1.4 | Data shard race condition | Client polls `/status` until `data_distributing`/`active` before download | `client/client.py` |
| 1.5 | Missing coordinator deps | Added `pandas>=1.3.0`, `scikit-learn>=0.24.0` to requirements.txt | `coordinator/requirements.txt` |

### Updated State Machine

```
[waiting_for_clients] → [data_distributing] → [active] ⇄ [aggregating]
                                                              ↓
                                                           [done]
```
After round 1, subsequent rounds transition directly to `active` (data already distributed).

### New Function: `credits.ensure_round_exists()`

Pre-creates a placeholder round row via `INSERT OR IGNORE` to satisfy the FK constraint on `credits(round) → rounds(round_number)`. Called on each `/submit` before `log_credit()`.

### Verification

- Integration tests: 6/6 passed
- Credits sanity tests: 14/14 passed
- All modified files pass syntax validation
