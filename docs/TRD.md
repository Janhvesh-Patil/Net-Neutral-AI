# Technical Requirements Document (TRD)
# Net-Neutral AI — Federated Learning Platform

**Version:** 1.0  
**Date:** 2026-07-28  
**Authors:** Janhvesh Patil, Tejas Kolekar, Atharv Huilgol, Bhoomika Salunke  
**Status:** Active Development (Prototype)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [Component Specifications](#3-component-specifications)
   - 3.1 [Coordinator Server](#31-coordinator-server-backendcoordinatorserverpy)
   - 3.2 [Client Agent](#32-client-agent-backendclientclientpy)
   - 3.3 [ML Model](#33-ml-model-backendclientmodelpy)
   - 3.4 [FedAvg Engine](#34-fedavg-engine-backendcoordinatorfedavgpy)
   - 3.5 [Evaluation Module](#35-evaluation-module-backendcoordinatorevaluatepy)
   - 3.6 [Credit Ledger](#36-credit-ledger-backendcoordinatorcreditspy)
   - 3.7 [Data Distributor](#37-data-distributor-backendcoordinatordata_distributorpy)
   - 3.8 [Frontend Dashboard](#38-frontend-dashboard-frontend)
   - 3.9 [Shared Configuration](#39-shared-configuration-backendsharedconfigpy)
4. [Database Schema](#4-database-schema)
5. [API Reference](#5-api-reference)
6. [Data Flow & State Machine](#6-data-flow--state-machine)
7. [ML Pipeline Specification](#7-ml-pipeline-specification)
8. [Networking & Communication Protocol](#8-networking--communication-protocol)
9. [Thread Safety & Concurrency Model](#9-thread-safety--concurrency-model)
10. [Memory Management](#10-memory-management)
11. [Configuration Reference](#11-configuration-reference)
12. [Dependency Manifests](#12-dependency-manifests)
13. [Error Handling & Failure Modes](#13-error-handling--failure-modes)
14. [Testing Architecture](#14-testing-architecture)
15. [Deployment Specifications](#15-deployment-specifications)

---

## 1. System Overview

Net-Neutral AI implements the **Federated Averaging (FedAvg)** algorithm (McMahan et al., 2017) to enable collaborative AI model training without centralising raw data.

### Core Architecture Pattern

```
                    ┌─────────────────────────┐
                    │    Web Frontend UI      │
                    │  (Real-Time Dashboard)  │
                    └────────────▲────────────┘
                                 │ SSE / HTTP
                    ┌────────────▼────────────┐
                    │   Coordinator Server    │
                    │     (Flask Backend)     │
                    │                         │
                    │  ┌─────────┐ ┌────────┐ │
                    │  │ FedAvg  │ │Credits │ │
                    │  │ Engine  │ │ SQLite │ │
                    │  └─────────┘ └────────┘ │
                    └────────────┬────────────┘
                                 │ Local WiFi or Cloud URL
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼──────┐     ┌─────────▼──────┐     ┌─────────▼──────┐
│   Client A     │     │   Client B     │     │   Client C     │
│                │     │                │     │                │
│  Local Shard   │     │  Local Shard   │     │  Local Shard   │
│  stays here    │     │  stays here    │     │  stays here    │
└────────────────┘     └────────────────┘     └────────────────┘

↑ Weight updates (what model learned) travel UP to coordinator
↓ Global model (averaged weights)     travel DOWN to clients
✗ Raw data never leaves any device
```

### Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| ML Framework | PyTorch | 2.3 |
| Model Architecture | Custom 2-layer Transformer | - |
| Coordinator Server | Flask + Gunicorn (production) | 3.0 / 23.x |
| Real-time Push | Server-Sent Events (SSE) | W3C standard |
| Database | SQLite 3 | built-in |
| Client Networking | Python `requests` | 2.x |
| Frontend | HTML5 / ES6 JavaScript / Vanilla CSS3 | - |
| Python Version | CPython | 3.12 |

---

## 2. Architecture Design

### 2.1 Architectural Style

The system uses a **Star Topology Federated Learning** architecture:
- **One Coordinator (Hub)**: Manages the global state machine, performs aggregation and evaluation, hosts the dataset shard distribution and the web dashboard.
- **N Clients (Spokes)**: Each client maintains a private data shard and performs local model training. Maximum N=3 in the current prototype.

### 2.2 Communication Patterns

| Channel | Protocol | Direction | Purpose |
|---------|----------|-----------|---------|
| Client → Coordinator | HTTP/REST (JSON + multipart) | Unidirectional | Registration, weight submission, epoch reporting |
| Coordinator → Client | HTTP/REST (JSON + binary) | Unidirectional | Model download, data shard delivery, status polling |
| Browser → Coordinator | HTTP GET (SSE) | Long-lived stream | Real-time event streaming |
| Browser → Coordinator | HTTP/REST | Bidirectional | Dashboard control (start training, upload dataset) |

### 2.3 State Machine

The coordinator implements a global state machine with the following states:

```
waiting_for_clients
       │
       │ (start_training called, all clients registered)
       ▼
data_distributing
       │
       │ (first client downloads shard)
       ▼
active  ◄──────────────────────────────────┐
       │                                   │
       │ (all clients submit weights)       │
       ▼                                   │
aggregating                                │
       │                                   │
       │ (FedAvg complete, evaluate)        │
       ▼                                   │
       if current_round < TOTAL_ROUNDS ────┘ (current_round += 1)
       │
       │ (current_round == TOTAL_ROUNDS)
       ▼
done
```

---

## 3. Component Specifications

### 3.1 Coordinator Server (`backend/coordinator/server.py`)

**Technology:** Flask 3.0 + Gunicorn (production WSGI)  
**Responsibility:** Central orchestrator — hosts the web UI, manages API endpoints, runs the federated training state machine.

#### Global State Variables

| Variable | Type | Description |
|----------|------|-------------|
| `current_round` | `int` | Current federated round number (1-indexed) |
| `TOTAL_ROUNDS` | `int` | Configurable total rounds (default: 5) |
| `LOCAL_EPOCHS` | `int` | Configurable local epochs per round (default: 2) |
| `registered_clients` | `set[str]` | Set of registered client IDs |
| `round_status` | `str` | Current state machine status |
| `global_accuracy` | `float` | Latest global model accuracy (0.0–1.0) |
| `accuracy_history` | `list[dict]` | `[{round, accuracy}]` per-round history |
| `submitted_weights` | `dict[str, str]` | `{client_id: filepath}` — pending weight files |
| `submitted_samples` | `dict[str, int]` | `{client_id: sample_count}` — for weighted FedAvg |
| `epoch_reports` | `dict[str, list]` | `{client_id: [{epoch, loss, accuracy, samples, round}]}` |
| `client_micro_states` | `dict[str, str]` | `{client_id: state}` for UI animation |
| `data_shards` | `dict[str, DataFrame]` | In-memory shard cache post-distribution |
| `sse_clients` | `list[Queue]` | One `Queue` per connected browser for SSE |
| `state_lock` | `threading.RLock` | Re-entrant lock for all state mutations |
| `sse_lock` | `threading.Lock` | Lock for SSE client list mutations |

#### Key Functions

| Function | Description |
|----------|-------------|
| `broadcast_event(event_type, data)` | Pushes an SSE payload to all connected browser queues. Automatically removes dead queues. |
| `run_evaluation_from_path(model_path, round_number)` | Loads a checkpoint, runs evaluation on the cached validation DataLoader, returns accuracy float, then frees memory. |
| `_process_round_completion(...)` | Background thread: runs FedAvg, saves global model, evaluates, logs credits, advances state machine. |
| `check_round_completion()` | Called after each weight submission — triggers `_process_round_completion` in a daemon thread when all clients have submitted. |

#### Static File Serving
The Flask app is configured with:
```python
app = Flask(
    __name__,
    static_folder=os.path.join(..., 'frontend'),
    static_url_path=''
)
```
This serves the frontend `index.html` at the root URL `/`.

---

### 3.2 Client Agent (`backend/client/client.py`)

**Technology:** Python 3.12 + `requests` library  
**Responsibility:** Connects to coordinator, downloads data shard + model, trains locally, uploads weights.

#### Client Execution Flow (One Round)

```python
1. Register with coordinator (POST /register)
2. GET /api/config  → fetch TOTAL_ROUNDS, LOCAL_EPOCHS
3. POST /get_data_shard  → download CSV shard to local_data/{client_id}_data.csv
4. GET /vocab  → download vocab.json
5. Poll GET /status every POLL_INTERVAL_SECS until round_status == 'active'
6. GET /model  → download checkpoint.pt to temp directory
7. POST /api/client_state {"state": "training"}  → notify dashboard
8. Run local training (model.py + train.py) for LOCAL_EPOCHS epochs
   - After each epoch: POST /report_epoch {epoch, loss, accuracy, samples, round}
9. POST /api/client_state {"state": "uploading"}
10. Compress weights with GZIP
11. POST /submit  → multipart upload of weights.pt.gz + metadata
    - Response: {credits, round, global_acc}
12. POST /api/client_state {"state": "idle"}
13. Poll for next round → repeat from step 6
```

#### Key Parameters

| Parameter | CLI Argument | Config Fallback | Default |
|-----------|-------------|-----------------|---------|
| Client ID | `--client_id` | `config.CLIENT_ID` | `"client_A"` |
| Coordinator URL | `--coordinator_url` | `config.BASE_URL` | `http://127.0.0.1:5000` |
| Total Rounds | fetched from `/api/config` | `config.TOTAL_ROUNDS` | 5 |
| Local Epochs | fetched from `/api/config` | `config.LOCAL_EPOCHS` | 2 |

---

### 3.3 ML Model (`backend/client/model.py`)

**Architecture:** `TransformerClassifier` — a 2-layer Transformer Encoder for binary text classification.

#### Model Architecture

```
Input: LongTensor (batch_size, seq_len)   [token IDs, padded to max_len=128]
  │
  ├─► Token Embedding (vocab_size=10000, embed_dim=128, padding_idx=0)
  ├─► Positional Embedding (max_len=128, embed_dim=128)  [learned, not sinusoidal]
  │
  x = dropout(token_embed + position_embed)
  │
  ├─► Padding Mask (True where input_ids == 0)
  │
  ├─► TransformerEncoder
  │     └─► TransformerEncoderLayer × 2
  │           d_model=128, nhead=4, dim_feedforward=256, dropout=0.1
  │           batch_first=True
  │
  x = GlobalAveragePooling (masked — ignores padding tokens)
  │
  ├─► Dropout (0.1)
  │
  └─► Linear Classifier (128 → 2)

Output: FloatTensor (batch_size, 2)  [raw logits for CrossEntropyLoss]
```

#### Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 10,000 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| FFN hidden dimension | 256 |
| Transformer layers | 2 |
| Max sequence length | 128 tokens |
| Dropout rate | 0.1 |
| Output classes | 2 (binary) |
| Total parameters | ~1,600,642 |

#### Weight Initialisation
- Classifier layer: Xavier Uniform for weights, zeros for bias.
- Embeddings: default PyTorch (Normal distribution).

---

### 3.4 FedAvg Engine (`backend/coordinator/fedavg.py`)

**Responsibility:** Implements Weighted Federated Averaging — aggregates client weight updates into a new global model.

#### Algorithm: Weighted FedAvg

```
For each layer key k in the global model:
    global[k] = Σ (client_i[k] × (samples_i / total_samples))
              for all valid client submissions
```

Where `total_samples = Σ samples_i` over all valid clients.

If `client_samples` is `None`, equal weighting (1 per client) is used.

#### Validation Pipeline (Pre-Aggregation)

Each client weight submission is validated before inclusion:

| Check | Condition | Action on Failure |
|-------|-----------|-------------------|
| Empty state_dict | `len(state_dict) == 0` | Skip client |
| Non-tensor values | Any value not a `torch.Tensor` | Skip client |
| Key mismatch | Keys differ from reference state_dict | Skip client |
| NaN check | `torch.isnan(val).any()` for any layer | Skip client |
| Inf check | `torch.isinf(val).any()` for any layer | Skip client |

Skipped clients are logged. If **all** clients fail validation, a `FedAvgError` is raised.

#### Result Container

```python
@dataclass
class FedAvgResult:
    global_state_dict:   Dict[str, torch.Tensor]  # The averaged model
    clients_included:    int                        # How many clients contributed
    included_client_ids: List[str]                  # Which client IDs
    skipped:             Dict[str, str]             # {client_id: reason} for exclusions
```

#### File Helpers

| Function | Description |
|----------|-------------|
| `load_client_weights(file_path, client_id)` | Safely loads a `.pt` state_dict to CPU. Returns `(state_dict, None)` on success, `(None, error_str)` on failure. |
| `save_global_model(state_dict, checkpoint_path)` | Writes averaged state_dict to `checkpoint.pt`. Logs file size. |
| `load_global_model(checkpoint_path)` | Loads `checkpoint.pt` to CPU state_dict. Raises `FileNotFoundError` if missing. |

---

### 3.5 Evaluation Module (`backend/coordinator/evaluate.py`)

**Responsibility:** Evaluates the global model on the coordinator's held-out validation set after each round.

#### Evaluation Metrics

All metrics are computed from raw prediction/label arrays without scikit-learn dependency:

| Metric | Formula |
|--------|---------|
| Accuracy | `correct / total` |
| Precision (per-class) | `tp[c] / (tp[c] + fp[c])` |
| Recall (per-class) | `tp[c] / (tp[c] + fn[c])` |
| F1 (per-class) | `2 * precision[c] * recall[c] / (precision[c] + recall[c])` |
| Macro Precision | `mean(precision[0], precision[1])` |
| Macro Recall | `mean(recall[0], recall[1])` |
| Macro F1 | `mean(f1[0], f1[1])` |
| Average Loss | `total_loss / num_batches` (CrossEntropyLoss) |

#### Result Container

```python
@dataclass
class EvalResult:
    accuracy:        float
    precision:       Dict[int, float]   # {0: float, 1: float}
    recall:          Dict[int, float]
    f1:              Dict[int, float]
    macro_precision: float
    macro_recall:    float
    macro_f1:        float
    support:         Dict[int, int]     # {class: sample_count}
    avg_loss:        float
    total_samples:   int
    correct:         int
    class_names:     Dict[int, str]     # {0: "negative", 1: "positive"}
```

#### Memory Management
After evaluation, the model is deleted and `gc.collect()` is called to keep peak memory within the 512 MB Render constraint.

---

### 3.6 Credit Ledger (`backend/coordinator/credits.py`)

**Technology:** SQLite 3 (Python `sqlite3` module)  
**Responsibility:** Persists all client registrations, credit transactions, and round metadata.

#### Database Records

```python
@dataclass
class ClientRecord:
    id, client_id, ip_address, registered_at, last_seen, is_active

@dataclass
class CreditRecord:
    id, client_id, round, samples_trained, time_seconds, points_earned, timestamp

@dataclass
class RoundRecord:
    id, round_number, started_at, completed_at, clients_submitted, global_accuracy, accuracy_delta

@dataclass
class LeaderboardEntry:
    rank, client_id, total_points, total_samples, rounds_participated
```

#### Credit Formula

```python
def compute_points(samples_trained: int) -> int:
    return samples_trained // 5
```

1 point is awarded for every 5 training samples processed by the client.

#### Key Functions

| Function | Description |
|----------|-------------|
| `init_db()` | Creates all tables if they don't exist. Sets journal_mode=DELETE. |
| `register_client(client_id, ip_address)` | INSERT OR IGNORE into `clients` table. |
| `log_credit(client_id, round, samples, time_seconds)` | Records a credit transaction, returns `points_earned`. |
| `log_round(round_num, started_at, clients_included, accuracy)` | Records a completed round. |
| `get_leaderboard()` | Returns `list[LeaderboardEntry]` sorted by total_points DESC. |
| `get_leaderboard_dicts()` | Returns `list[dict]` for JSON serialisation. |
| `get_stats()` | Returns aggregate stats: total_clients, total_samples, total_rounds. |
| `ensure_round_exists(round_num, started_at)` | Creates a round row if missing (prevents FK constraint violations). |

#### SQLite Configuration

```sql
PRAGMA journal_mode=DELETE;   -- Safe on ephemeral filesystems (Render)
PRAGMA foreign_keys = ON;     -- Enforce referential integrity
```

> **Note:** WAL mode is explicitly avoided because `.db-wal` sidecar files can become orphaned on Render's ephemeral filesystem, causing silent database corruption.

---

### 3.7 Data Distributor (`backend/coordinator/data_distributor.py`)

**Responsibility:** Loads, validates, and shards the uploaded CSV dataset.

#### Key Functions

| Function | Description |
|----------|-------------|
| `load_and_validate_csv(path)` | Reads CSV, asserts `review` and `label` columns exist, validates `label` is binary. Returns DataFrame. |
| `divide_dataset(path, num_clients)` | 80/20 split. Saves test set to `uploaded_test.csv`. Shards 80% equally among `num_clients`. Returns `{client_id: DataFrame}`. |
| `validate_shards(shards, total_samples)` | Asserts no shard is empty and total sample count is correct. |

#### Shard Naming Convention

Shards are assigned in order of `registered_clients` iteration:
```
client_A → shard 0   (rows 0..N/num_clients)
client_B → shard 1
client_C → shard 2
```

---

### 3.8 Frontend Dashboard (`frontend/`)

**Files:**
- `index.html` — Single-page application shell (~1,300 lines)
- `styles.css` — Glassmorphic dark theme styling (~1,000 lines)
- `app.js` — Core event handling, SSE client, SVG topology renderer (~2,200 lines)

#### JavaScript Architecture

| Module Area | Responsibility |
|-------------|----------------|
| SSE Client | Connects to `/events`, dispatches incoming events to handlers |
| Polling Fallback | Falls back to `GET /status` polling if SSE unavailable |
| Topology SVG | Renders coordinator and client nodes as animated SVG elements |
| Chart Rendering | Draws accuracy history chart |
| Epoch Tables | Maintains tabbed per-round epoch data tables |
| Leaderboard | Renders sorted credit table |
| Timeline | Reconstructs client timeline from server history API |
| Session Manager | Handles dataset upload, training start, model download |

#### SSE Event Handlers

```javascript
eventSource.addEventListener('client_joined',   handleClientJoined);
eventSource.addEventListener('epoch_update',    handleEpochUpdate);
eventSource.addEventListener('accuracy_update', handleAccuracyUpdate);
eventSource.addEventListener('round_start',     handleRoundStart);
eventSource.addEventListener('sys_log',         handleSysLog);
eventSource.addEventListener('training_done',   handleTrainingDone);
eventSource.addEventListener('session_cleanup', handleSessionCleanup);
```

---

### 3.9 Shared Configuration (`backend/shared/config.py`)

All configurable constants in one place, with environment variable overrides:

```python
# Network
COORDINATOR_IP   = os.environ.get('COORDINATOR_IP', '127.0.0.1')
COORDINATOR_PORT = int(os.environ.get('COORDINATOR_PORT', 5000))
BASE_URL         = os.environ.get('COORDINATOR_BASE_URL', f"http://{COORDINATOR_IP}:{COORDINATOR_PORT}")

# Training
TOTAL_ROUNDS  = 5
LOCAL_EPOCHS  = 2
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3

# Timeouts
SUBMISSION_TIMEOUT_SECS    = 90
POLL_INTERVAL_SECS         = 5
REGISTER_RETRY_ATTEMPTS    = 3
REGISTER_RETRY_DELAY       = 10
WAIT_FOR_DATA_TIMEOUT_SECS = 300

# Data
LOCAL_DATA_DIR      = "local_data"
DATA_SHARD_FILENAME = "{client_id}_data.csv"
CHECKPOINT_FILENAME = "checkpoint.pt"
DB_FILENAME         = "database.db"
MAX_CLIENTS         = 3
```

---

## 4. Database Schema

**Database File:** `backend/coordinator/database.db` (SQLite 3)

```sql
-- Client registry (persistent)
CREATE TABLE IF NOT EXISTS clients (
    id            INTEGER  PRIMARY KEY AUTOINCREMENT,
    client_id     TEXT     NOT NULL UNIQUE,
    ip_address    TEXT     NOT NULL DEFAULT 'unknown',
    registered_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen     DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active     INTEGER  NOT NULL DEFAULT 1
);

-- Per-round metadata (persistent)
CREATE TABLE IF NOT EXISTS rounds (
    id                 INTEGER  PRIMARY KEY AUTOINCREMENT,
    round_number       INTEGER  NOT NULL UNIQUE,
    started_at         DATETIME NOT NULL,
    completed_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    clients_submitted  INTEGER  NOT NULL DEFAULT 0,
    global_accuracy    REAL     NOT NULL DEFAULT 0.0,
    accuracy_delta     REAL     DEFAULT 0.0
);

-- Credit transactions (persistent)
CREATE TABLE IF NOT EXISTS credits (
    id               INTEGER  PRIMARY KEY AUTOINCREMENT,
    client_id        TEXT     NOT NULL REFERENCES clients(client_id),
    round            INTEGER  NOT NULL REFERENCES rounds(round_number),
    samples_trained  INTEGER  NOT NULL DEFAULT 0,
    time_seconds     REAL     NOT NULL DEFAULT 0.0,
    points_earned    INTEGER  NOT NULL DEFAULT 0,
    timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes (Implicit)

| Table | Index | Purpose |
|-------|-------|---------|
| `clients` | `UNIQUE(client_id)` | Prevent duplicate registration |
| `rounds` | `UNIQUE(round_number)` | Prevent duplicate round entries |
| `credits` | FK `client_id` → `clients` | Referential integrity |
| `credits` | FK `round` → `rounds` | Referential integrity |

---

## 5. API Reference

### Base URL
- **Local:** `http://localhost:5000`
- **Cloud:** `https://your-app.onrender.com`

### Endpoints

#### `POST /register`
Register a client node with the coordinator.

**Request Body (JSON):**
```json
{
  "client_id": "client_A",
  "ip_address": "192.168.1.42"
}
```

**Response (200):**
```json
{"status": "ok", "round": 1}
```

**SSE Side Effect:** Broadcasts `client_joined` event.

---

#### `GET /model`
Download the current global model checkpoint.

**Response:** Binary `.pt` file (PyTorch state dict, ~6 MB)

---

#### `POST /submit`
Submit locally trained weights after a round.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `client_id` | string | Client identifier |
| `weights` | file | `.pt` or `.pt.gz` weight file |
| `samples_trained` | int | Number of training samples processed |
| `time_seconds` | float | Local training duration |

**Response (200):**
```json
{"credits": 120, "round": 1, "global_acc": 0.832}
```

---

#### `GET /status`
Poll coordinator state (used by clients between rounds).

**Response (200):**
```json
{
  "round": 2,
  "round_status": "active",
  "active_clients": ["client_A", "client_B", "client_C"],
  "clients_submitted": ["client_A"],
  "global_accuracy": 0.832,
  "accuracy_history": [{"round": 1, "accuracy": 0.832}],
  "total_rounds": 5,
  "leaderboard": [...],
  "client_micro_states": {"client_A": "idle"},
  "epoch_reports": {"client_A": [...]},
  "clients": [{"id": "client_A", "ip": "192.168.1.42"}]
}
```

---

#### `POST /upload_dataset`
Upload a CSV dataset file (Coordinator dashboard only).

**Request:** `multipart/form-data` with `file` field (`.csv` only)

**Response (200):**
```json
{"status": "ok", "rows": 10000}
```

**Response (400):** `{"error": "File must be CSV format"}`

---

#### `POST /start_training`
Begin data distribution and federated training session.

**Request Body (JSON):**
```json
{"client_count": 3, "rounds": 5, "epochs": 2}
```

**Response (200):**
```json
{"status": "ok", "shards_prepared": 3}
```

**Side Effect:** Resets all global state, divides dataset, transitions `round_status → data_distributing`.

---

#### `POST /get_data_shard`
Client requests its private training data shard.

**Request Body (JSON):**
```json
{"client_id": "client_A"}
```

**Response (200):** Binary CSV file (`data_shard.csv`)

---

#### `GET /vocab`
Download the global vocabulary JSON.

**Response (200):** `vocab.json` (JSON, `application/json`)

---

#### `GET /api/config`
Fetch training hyperparameters.

**Response (200):**
```json
{
  "total_rounds": 5,
  "local_epochs": 2,
  "epochs": 2,
  "batch_size": 32,
  "learning_rate": 0.001
}
```

---

#### `GET /api/client_status/<client_id>`
Fetch individual client's state and round history.

**Response (200):**
```json
{
  "client_id": "client_A",
  "ip_address": "192.168.1.42",
  "registered_at": "2026-07-28T12:00:00",
  "total_credits": 600,
  "current_round": 2,
  "round_status": "active",
  "micro_status": "idle",
  "total_rounds": 5,
  "has_submitted_this_round": false,
  "global_accuracy": 0.832,
  "round_history": [...],
  "epoch_reports": [...]
}
```

---

#### `POST /api/client_state`
Update a client's micro-state for UI animations.

**Request Body (JSON):**
```json
{"client_id": "client_A", "state": "training"}
```

Valid states: `"idle"`, `"downloading"`, `"training"`, `"uploading"`

---

#### `POST /report_epoch`
Client reports per-epoch training metrics.

**Request Body (JSON):**
```json
{
  "client_id": "client_A",
  "epoch": 1,
  "loss": 0.432,
  "accuracy": 0.821,
  "samples": 2667,
  "round": 1
}
```

**SSE Side Effect:** Broadcasts `epoch_update` event.

---

#### `GET /events`
Server-Sent Events stream for real-time dashboard updates.

**Response:** `text/event-stream` (long-lived HTTP connection)

SSE event format:
```
event: <event_type>
data: <JSON payload>

```

Keepalive is sent every 20 seconds:
```
: keepalive

```

---

#### `GET /download_model`
Download trained model and trigger session cleanup.

**Response:** Binary `.pt` file named `net_neutral_trained_model.pt`

**Side Effect:** Schedules `cleanup_session_data()` in a daemon thread 5 seconds after response.

---

#### `GET /leaderboard`
Public credit leaderboard.

**Response (200):**
```json
{
  "leaderboard": [
    {"rank": 1, "client_id": "client_A", "total_points": 600, "total_samples": 3000, "rounds_participated": 5}
  ]
}
```

---

#### `GET /stats`
Aggregate platform statistics.

**Response (200):**
```json
{"total_clients": 3, "total_samples": 9000, "total_rounds": 5}
```

---

#### `GET /results`
Final training results.

**Response (200):**
```json
{"final_accuracy": 0.891, "total_rounds_completed": 5}
```

---

## 6. Data Flow & State Machine

### Round Lifecycle (Detailed)

```
Phase 1 — Setup:
  Browser: POST /upload_dataset → validate CSV → build vocab → save test split
  Browser: POST /start_training {client_count, rounds, epochs}
           → reset all state → shard data → status: 'data_distributing'

Phase 2 — Client Onboarding:
  Client: POST /register → persist to DB → broadcast 'client_joined' SSE
  Client: POST /get_data_shard → serve CSV shard → status: 'active'
  Client: GET /vocab → serve vocab.json

Phase 3 — Training Round (repeated TOTAL_ROUNDS times):
  Client: Poll GET /status until round_status == 'active' and round == N
  Client: GET /model → download checkpoint.pt
  Client: POST /api/client_state {"state": "training"}
  Client: [Local Training Loop for LOCAL_EPOCHS epochs]
    → After each epoch: POST /report_epoch → SSE 'epoch_update'
  Client: POST /api/client_state {"state": "uploading"}
  Client: POST /submit {weights.pt.gz, samples_trained, time_seconds}
          → decompress GZIP → save temp file
          → log credits to SQLite
          → call check_round_completion()

  [Background Thread — when all clients submitted]:
    → Load all temp weight files
    → Run federated_average() → FedAvgResult
    → Save global model to checkpoint.pt
    → Run evaluation on validation set → accuracy
    → log_round() to SQLite
    → Broadcast 'accuracy_update' SSE
    → Delete temp weight files
    → if round < TOTAL_ROUNDS: current_round += 1, status: 'active', broadcast 'round_start'
    → if round == TOTAL_ROUNDS: status: 'done', broadcast 'training_done'

Phase 4 — Export:
  Browser: GET /download_model → serve checkpoint.pt
           → 5s delay → cleanup_session_data()
           → restore checkpoint_pretrained_backup.pt → broadcast 'session_cleanup'
```

---

## 7. ML Pipeline Specification

### Tokenisation

```python
class Vocabulary:
    PAD = 0  # padding token
    UNK = 1  # unknown token
    VOCAB_SIZE = 10_000
    TRAIN_SIZE = 8_000  # max training samples for vocab building

    def build(texts, max_size=VOCAB_SIZE):
        # Count word frequencies in all texts
        # Keep top max_size-2 tokens (reserve 0=PAD, 1=UNK)
        # Save as {word: int_id} JSON

    def encode(text, max_len=128):
        # Lowercase → split on whitespace/punctuation
        # Map to token IDs (1=UNK for OOV)
        # Truncate to max_len
        # Return list of ints
```

### Data Loading

```python
class TextDataset(Dataset):
    # (encoded_tensor, label_tensor) pairs
    # encoded_tensor: shape (max_len,) = (128,)
    # label_tensor: scalar int64

def collate_fn(batch):
    # Pad all sequences in batch to the longest sequence
    # Returns (batch_ids, batch_labels) — shape (B, max_len) and (B,)
```

### Local Training

```python
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = CrossEntropyLoss()

for epoch in range(LOCAL_EPOCHS):
    for batch_ids, batch_labels in dataloader:
        optimizer.zero_grad()
        logits = model(batch_ids)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
    # Report epoch metrics to coordinator
```

### Pre-Training Baseline

The `checkpoint.pt` in this repo was generated by `pretrain.py`:
- Trains on a clean sentiment dataset with full vocabulary.
- 5 epochs, Adam optimizer, lr=1e-3.
- Expected validation accuracy: ~80.95%.
- Serves as the starting point for all federated rounds.

---

## 8. Networking & Communication Protocol

### Weight Compression (Client → Coordinator)

```python
# Client-side compression
import gzip, io, torch
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
compressed = gzip.compress(buffer.getvalue())
# Upload as multipart with filename='weights.pt.gz'
```

```python
# Coordinator-side decompression (server.py)
if raw_path.endswith('.gz'):
    with gzip.open(raw_path, 'rb') as f_in, open(save_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(raw_path)
```

### Model Download (Coordinator → Client)

Plain binary `.pt` file served via `Flask.send_file()`.  
Client saves to a temp directory, loads with:
```python
torch.load(path, map_location="cpu", weights_only=True)
```

### SSE Stream Protocol

```
Connection: Keep-Alive
Content-Type: text/event-stream
Cache-Control: no-cache

event: init
data: {"round": 1, "round_status": "active", ...}

event: client_joined
data: {"client_id": "client_A", "ip": "192.168.1.42", "total_clients": 1}

: keepalive        ← sent every 20s (blank comment line)
```

### LAN Discovery (`lan_scan.py`)

Scans the local `/24` subnet via ICMP or port 5000 probe to auto-discover the coordinator IP, removing the need to manually find the machine's IPv4 address.

---

## 9. Thread Safety & Concurrency Model

### Coordinator Threading Model (Gunicorn: `-w 1 --threads 4`)

Using **1 worker with 4 threads** is a critical design constraint:
- The global state (`current_round`, `registered_clients`, etc.) is stored in Python module-level variables.
- Multiple worker **processes** would not share this state (no shared memory by default in Python multiprocessing).
- Therefore, exactly **one process with multiple threads** is required.

### Locks

| Lock | Type | Scope |
|------|------|-------|
| `state_lock` | `threading.RLock()` | All mutations to `current_round`, `round_status`, `submitted_weights`, `submitted_samples`, `accuracy_history` |
| `sse_lock` | `threading.Lock()` | All reads/writes to `sse_clients` list |

### Background Thread Pattern

FedAvg aggregation is always run in a **daemon background thread** to prevent blocking the main Flask worker:

```python
threading.Thread(
    target=_process_round_completion,
    args=(round_copy, time_copy, weights_copy, samples_copy),
    daemon=True
).start()
```

State is **snapshot-copied** before launching the thread so the main thread can immediately clear `submitted_weights` for the next round without a race condition.

---

## 10. Memory Management

### Coordinator Memory Budget (Render Free Tier: 512 MB)

| Object | Estimated Size | Lifecycle |
|--------|---------------|-----------|
| Global model (`checkpoint.pt`) | ~6 MB (in-memory tensor) | Loaded per-evaluation, then deleted |
| Validation DataLoader | ~10 MB | Cached after first evaluation (`_cached_val_loader`) |
| Uploaded dataset CSV | Variable | Deleted after training completes |
| Data shards (in-memory) | Variable | Kept until training starts, then served and cleared |
| Per-client weight files | ~6 MB each | Saved to disk, aggregated, then deleted |

### Post-Download Cleanup (`cleanup_session_data`)

Executed 5 seconds after model download:

1. Delete `uploaded_dataset.csv`
2. Delete `uploaded_test.csv`
3. Delete all `temp_*_round*.pt` files (glob)
4. Restore `checkpoint.pt` from `checkpoint_pretrained_backup.pt`
5. Clear `data_shards` dict (in-memory)
6. Clear `epoch_reports` dict (in-memory)
7. Broadcast `session_cleanup` SSE event

---

## 11. Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COORDINATOR_IP` | Coordinator's IPv4 address | `127.0.0.1` |
| `COORDINATOR_PORT` | Flask server port | `5000` |
| `COORDINATOR_BASE_URL` | Full base URL override | Computed from IP + PORT |
| `PORT` | Render auto-sets this; Gunicorn binds to it | `5000` |

### Training Hyperparameters (`shared/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOTAL_ROUNDS` | 5 | Number of federated rounds |
| `LOCAL_EPOCHS` | 2 | Local training epochs per round |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `LEARNING_RATE` | 1e-3 | AdamW learning rate |

### Timeout Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SUBMISSION_TIMEOUT_SECS` | 90 | Max wait for all clients to submit |
| `POLL_INTERVAL_SECS` | 5 | Client polling frequency |
| `REGISTER_RETRY_ATTEMPTS` | 3 | Max registration retry attempts |
| `REGISTER_RETRY_DELAY` | 10 | Seconds between retries |
| `WAIT_FOR_DATA_TIMEOUT_SECS` | 300 | Max wait for data shard |

---

## 12. Dependency Manifests

### Coordinator (`backend/coordinator/requirements.txt`)
```
flask
flask-cors
torch
pandas
```

### Coordinator — Render Production (`backend/coordinator/requirements-render.txt`)
```
flask
flask-cors
torch --index-url https://download.pytorch.org/whl/cpu
gunicorn
pandas
```

> Uses CPU-only PyTorch wheel to reduce Docker image size and memory footprint on Render.

### Client (`backend/client/requirements.txt`)
```
torch
requests
pandas
```

### Client — CPU Fallback (`backend/client/requirements-cpu.txt`)
```
torch --index-url https://download.pytorch.org/whl/cpu
requests
pandas
```

---

## 13. Error Handling & Failure Modes

### Client Failures

| Failure | Handling |
|---------|----------|
| Registration timeout | Retry up to `REGISTER_RETRY_ATTEMPTS` times with `REGISTER_RETRY_DELAY` backoff |
| Data shard not available | Poll with `WAIT_FOR_DATA_TIMEOUT_SECS` timeout, then exit with error |
| Model download fails | Log error, skip round |
| Local training exception | Log error, skip weight submission for this round |
| Weight upload fails | Log error; coordinator times out this client after `SUBMISSION_TIMEOUT_SECS` |

### Coordinator Failures

| Failure | Handling |
|---------|----------|
| Corrupted client weights | Skip client in FedAvg, log warning, continue with valid submissions |
| All client weights invalid | `FedAvgError` raised, logged; round may not advance |
| Evaluation exception | Logged, `global_accuracy` unchanged, round still advances |
| Thread lock contention | `RLock` is re-entrant, preventing deadlock within same thread |
| SSE client disconnect | Dead queues detected on next broadcast; removed from `sse_clients` |
| File write error (COORDINATOR_DIR not writable) | `RuntimeError` raised at import time; server fails to start |

### Race Condition Mitigations

| Race | Mitigation |
|------|------------|
| Two clients download shard simultaneously (BUG-03) | `round_status='active'` also accepted for shard download (not only `data_distributing`) |
| Client submits weights while status is `data_distributing` (BUG-05) | Submit handler forcibly transitions status to `active` |
| State mutation mid-round in background thread | All mutations wrapped in `with state_lock:` |
| FK constraint violation (round row not created before credit) | `ensure_round_exists()` called before `log_credit()` |

---

## 14. Testing Architecture

### Integration Tests (`integration_test.py`)

Tests verify:
- Local IP address utility functions return valid IPs.
- Shared config values are within expected ranges.
- Required directory structure (`backend/client`, `backend/coordinator`, `frontend`) exists.
- `vocab.json` schema is valid (contains `word2idx`, `idx2word` keys).
- `checkpoint.pt` file is present and loadable.

### End-to-End Tests (`e2e_test.py`)

Simulates a complete federated session without real network:
- Starts a Flask test client.
- Uploads a small synthetic CSV dataset.
- Registers 3 virtual clients.
- Calls `/start_training`.
- Simulates shard downloads for all clients.
- Simulates local weight generation and `/submit` calls.
- Asserts `round_status == 'done'` after all rounds complete.
- Verifies accuracy was written to the database.

### Model Sanity Check (`backend/client/model.py` — `__main__`)

When run directly:
- Creates a `TransformerClassifier` instance.
- Runs a forward pass with a random batch `(8, 128)`.
- Asserts output shape is `(8, 2)`.
- Prints total parameter count (~1.6M).
- Prints the full `state_dict` key layout.

### FedAvg Sanity Check (`backend/coordinator/fedavg.py` — `__main__`)

7-test suite covering:
1. Equal samples → weighted == unweighted
2. Unequal samples → different from equal weighting
3. One client missing → partial aggregation
4. Corrupted client excluded → FedAvgError on NaN/key mismatch
5. All clients invalid → `FedAvgError` raised
6. Identical models → average equals input
7. Save/load round-trip integrity

### Pre-training Verification

```bash
python backend/coordinator/pretrain.py backend/data --verify-only
```

Expected: `✓ Checkpoint is demo-ready.` with accuracy ~80.95%.

---

## 15. Deployment Specifications

### Local Development

```bash
# Start coordinator
python backend/coordinator/server.py
# Binds to http://0.0.0.0:5000

# Start clients (separate terminals)
python backend/client/client.py --client_id client_A --coordinator_url http://localhost:5000
python backend/client/client.py --client_id client_B --coordinator_url http://localhost:5000
python backend/client/client.py --client_id client_C --coordinator_url http://localhost:5000
```

### Automated Demo (Windows)

```batch
:: backend/demo/run_demo.bat
:: Opens Windows Terminal with 4 panes:
::   Pane 0: coordinator
::   Pane 1: client_A
::   Pane 2: client_B
::   Pane 3: client_C
wt new-tab --title "Coordinator" cmd /k "python backend/coordinator/server.py"^
 ; split-pane --title "Client A"  cmd /k "python backend/client/client.py --client_id client_A"^
 ; split-pane --title "Client B"  cmd /k "python backend/client/client.py --client_id client_B"^
 ; split-pane --title "Client C"  cmd /k "python backend/client/client.py --client_id client_C"
```

### Cloud Deployment (Render.com)

```yaml
# Effective render.yaml equivalent:
service:
  type: web
  runtime: python
  buildCommand: pip install -r backend/coordinator/requirements-render.txt
  startCommand: >
    gunicorn
    -w 1
    --threads 4
    -b 0.0.0.0:$PORT
    --timeout 300
    backend.coordinator.server:app
  envVars:
    - key: PORT
      value: auto
```

**Critical constraints:**
- `--workers 1`: Required — multiple workers cannot share in-memory state.
- `--threads 4`: Needed to handle concurrent browser + client HTTP requests.
- `--timeout 300`: Required for large model uploads and downloads.

### GitHub Actions CI

The repository includes a `.github/` directory with a `lint.yml` workflow:
```yaml
# Runs Python linting on every push and pull request
# Badge: [![Lint](https://github.com/Janhvesh-Patil/net-neutral-ai/actions/workflows/lint.yml/badge.svg)]
```

---

## Appendix A: File Structure Reference

```
net-neutral-ai/
├── backend/
│   ├── client/
│   │   ├── client.py               # Client entry point — runs federated training loop
│   │   ├── model.py                # TransformerClassifier architecture
│   │   ├── train.py                # Local training loop (one round)
│   │   ├── data.py                 # Shard loading, tokenisation, DataLoader factory
│   │   ├── requirements.txt        # Client library dependencies
│   │   └── requirements-cpu.txt    # CPU-only fallback dependencies
│   ├── coordinator/
│   │   ├── server.py               # Flask coordinator — all API endpoints & web hosting
│   │   ├── fedavg.py               # Weighted FedAvg algorithm
│   │   ├── evaluate.py             # Evaluation module — accuracy, precision, recall, F1
│   │   ├── credits.py              # SQLite credit ledger read/write operations
│   │   ├── pretrain.py             # One-time baseline pre-training / verification script
│   │   ├── data_distributor.py     # CSV dataset loader, sharding, and validation
│   │   ├── lan_scan.py             # LAN scanner utility for discovering coordinator IP
│   │   ├── checkpoint.pt           # Current global model weights (~6MB)
│   │   ├── checkpoint_pretrained_backup.pt  # Initial baseline pre-trained checkpoint
│   │   ├── pretrain_log.json       # Pretraining metrics & metadata log
│   │   ├── vocab.json              # Global vocabulary (generated after dataset upload)
│   │   ├── requirements.txt        # Backend coordinator dependencies
│   │   └── requirements-render.txt # Production Render deployment dependencies
│   ├── shared/
│   │   ├── config.py               # Shared hyperparameters, ports, and configuration
│   │   └── ip_utils.py             # IP network helper scripts
│   └── demo/
│       └── run_demo.bat            # Windows Terminal launcher — 4 tiled terminals
├── frontend/
│   ├── index.html                  # Single-page Dashboard UI
│   ├── styles.css                  # Modern Glassmorphic styling
│   └── app.js                      # Core SSE & chart rendering script
├── docs/
│   ├── PRD.md                      # Product Requirements Document
│   └── TRD.md                      # Technical Requirements Document (this file)
├── local_data/                     # Directory for client data shards (generated at runtime)
├── integration_test.py             # Automated network and data integration tests
├── e2e_test.py                     # Fully simulated end-to-end local training test
├── Pretrained_Model.ipynb          # Jupyter notebook — model pre-training reference
├── FUTURE_SCOPES.md                # Planned future features and roadmap
├── WORKING.md                      # Architecture how-it-works documentation
├── RENDER_DEPLOYMENT.md            # Cloud deployment step-by-step guide
└── README.md                       # Project overview and quick-start guide
```

---

## Appendix B: Known Bugs & Fixes

| Bug ID | Description | Fix Applied |
|--------|-------------|-------------|
| BUG-03 | Race: second client calls `/get_data_shard` after status already transitions to `active` | Allow both `data_distributing` AND `active` in round 1 for shard endpoint |
| BUG-04 | Second training session on Render starts in dirty state (stale round, weights, etc.) | Full state reset in `/start_training` handler |
| BUG-05 | Client submits weights while status is `data_distributing`, blocking state machine | Submit handler forcibly sets status to `active` on receipt |
| BUG-06 | `KeyError` on missing `time_seconds` form field locking `state_lock` permanently | Use `.get()` with fallback instead of direct `form['time_seconds']` access |
| BUG-07 | WAL mode creates orphaned sidecar files on Render ephemeral storage | Switch SQLite to `journal_mode=DELETE` |
| BUG-16 | Cleanup deletes `checkpoint.pt` but WSGI module-level restore doesn't work reliably | Only restore if `_checkpoint_restored` flag is True |
| BUG-18 | SSE keepalive fired at 30s, within Render's 30s idle close window | Reduced to 20s keepalive interval |

---

*This document is maintained by the Net-Neutral AI team. For questions, see the [README](../README.md) or open a GitHub issue.*
