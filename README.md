# Net-Neutral AI

> _Democratising AI training — one idle GPU at a time._

[![Lint](https://github.com/Janhvesh-Patil/net-neutral-ai/actions/workflows/lint.yml/badge.svg)](https://github.com/Janhvesh-Patil/net-neutral-ai/actions/workflows/lint.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What is Net-Neutral AI?

Training a frontier AI model costs upwards of \$100M and is accessible to fewer than five organisations globally. Meanwhile, billions of devices — student laptops, gaming PCs, smartphones — sit idle with capable GPUs doing nothing.

**Net-Neutral AI is a federated learning platform** that recruits these idle devices as volunteer compute nodes, enabling collaborative AI model training without any single company controlling the infrastructure.

**Two guarantees by design:**

- Raw data never leaves a device — only learned weight updates travel across the network
- Every contributor is tracked and rewarded transparently through a persistent credit ledger

This is the net neutrality principle applied to AI compute: equal access, distributed power, no gatekeepers.

---

## What's New in v2.0

Since the initial prototype, the project has undergone a **major architectural overhaul**:

| Area                  | v1 (Prototype)                            | v2.0 (Current)                                                   |
| --------------------- | ----------------------------------------- | ---------------------------------------------------------------- |
| **Project layout**    | Flat `client/`, `coordinator/`, `shared/` | Organised under `backend/` with dedicated `frontend/`            |
| **Frontend**          | None — terminal only                      | Full 13-screen SPA with dark-theme mission-control UI            |
| **Data pipeline**     | Static local CSV shards                   | Dynamic upload → 80/20 split → stratified distribution           |
| **Real-time updates** | Polling only                              | Server-Sent Events (SSE) + polling fallback                      |
| **Deployment**        | Local WiFi only                           | Production WSGI server support (Gunicorn)                        |
| **Database**          | 2-table SQLite                            | 3-table SQLite with FK constraints, indexes, and `clients` table |
| **Weight transfer**   | Raw `.pt` upload                          | Gzip-compressed uploads with retry + decompression               |
| **Training config**   | Hardcoded in `config.py`                  | Dynamic rounds/epochs set from dashboard, synced to clients      |
| **Model download**    | Manual file copy                          | One-click download from results screen + auto-cleanup            |
| **Testing**           | None                                      | Integration tests, E2E tests, per-module sanity checks           |
| **Network discovery** | Manual IP entry                           | LAN subnet scanner + auto-detected `window.location.origin`      |

---

## System Architecture

```
                                    ┌──────────────────────────────────────┐
                                    │       Coordinator (Flask/Gunicorn)   │
                                    │                                      │
                                    │  ┌──────────┐  ┌─────────────────┐  │
                                    │  │ FedAvg   │  │ Credits (SQLite)│  │
                                    │  │ Engine   │  │ 3 tables + FK   │  │
                                    │  └──────────┘  └─────────────────┘  │
                                    │  ┌──────────┐  ┌─────────────────┐  │
                                    │  │ Data     │  │ SSE Broadcast   │  │
                                    │  │ Distrib. │  │ (Real-Time)     │  │
                                    │  └──────────┘  └─────────────────┘  │
                                    └─────────────────┬────────────────────┘
                                                      │
                    ┌────────────────── HTTP/SSE ──────┼────────────────────┐
                    │                                  │                    │
           ┌────────▼──────────┐            ┌──────────▼────────┐  ┌───────▼─────────┐
           │   Client A        │            │   Client B        │  │   Client C      │
           │                   │            │                   │  │                 │
           │  📥 Data shard    │            │  📥 Data shard    │  │  📥 Data shard  │
           │  🧠 Local model   │            │  🧠 Local model   │  │  🧠 Local model │
           │  💻 PyTorch train │            │  💻 PyTorch train │  │  💻 PyTorch     │
           │  stays here ✗     │            │  stays here ✗     │  │  stays here ✗   │
           └───────────────────┘            └───────────────────┘  └─────────────────┘

           ↑ Gzip-compressed weight updates travel UP to coordinator
           ↓ Global model (averaged weights) travel DOWN to clients
           ✗ Raw data never leaves any device
```

### One Round of Federated Training

1. **Coordinator** distributes the global model checkpoint to all clients
2. Each **client** trains locally on its own data shard for _N_ epochs
3. Clients submit updated weights (gzip-compressed) — **not raw data**
4. Coordinator runs **Weighted FedAvg** — averages all weight updates proportional to samples trained
5. Coordinator evaluates the new global model on a held-out 20% validation set
6. Credits are logged, leaderboard updates, and the next round begins
7. Repeat for the configured number of rounds (default 5)

---

## Web Dashboard

The frontend is a full single-page application with **13 screens**, built with vanilla HTML/CSS/JS and a dark-theme mission-control aesthetic:

| Screen                    | Description                                                                      |
| ------------------------- | -------------------------------------------------------------------------------- |
| **Landing Page**          | Hero section, live network stats, role-based entry (Coordinator / Client)        |
| **Sign In / Sign Up**     | Auth screens with role selection (Coordinator or Client Node)                    |
| **Coordinator Setup**     | Session name, expected client count, network configuration                       |
| **Coordinator Dashboard** | Connected clients table, training phase banner, live training log                |
| **New Training Job**      | 3-step wizard — upload CSV dataset, optional checkpoint, configure rounds/epochs |
| **Live Training View**    | 3-column mission control — nodes, accuracy chart, epoch metrics, system log      |
| **Job Results**           | Final accuracy, download trained model (.pt), download report (.json)            |
| **Client Setup**          | Session browser, client ID selector, connect button                              |
| **Client Discovery**      | Animated radar UI while establishing connection                                  |
| **Client Dashboard**      | Credits, training timeline, animated status visualiser, activity log             |
| **Agent Setup Guide**     | OS-specific CLI commands (Windows/macOS/Linux) for running the background agent  |
| **Leaderboard**           | Public lifetime leaderboard across all training sessions                         |

**Key frontend features:**

- **Server-Sent Events (SSE)** for zero-latency dashboard updates
- **Chart.js** real-time accuracy graph with round-over-round progression
- **Canvas network topology** visualising live connections between coordinator and clients
- **Animated status images** for each client micro-state (idle, downloading, training, uploading, aggregating)
- **Dynamic round/epoch tabs** with per-client loss and accuracy metrics

---

## Results

Baseline checkpoint — pre-trained on 15,000 samples for 5 epochs on GPU:

| Metric                   | Value                                    |
| ------------------------ | ---------------------------------------- |
| Best validation accuracy | **80.95%**                               |
| Correct / Total          | 1,619 / 2,000                            |
| Eval loss                | 0.5078                                   |
| Training time            | 40.0 seconds                             |
| Model parameters         | 1,561,602                                |
| Dataset                  | IMDb Movie Reviews (15K training subset) |

**Per-class breakdown (baseline checkpoint):**

| Class         | Precision  | Recall     | F1         | Support |
| ------------- | ---------- | ---------- | ---------- | ------- |
| Negative      | 0.8194     | 0.7870     | 0.8029     | 986     |
| Positive      | 0.8006     | 0.8314     | 0.8157     | 1,014   |
| **Macro avg** | **0.8100** | **0.8092** | **0.8093** | 2,000   |

---

## Tech Stack

| Component          | Technology                                                                |
| ------------------ | ------------------------------------------------------------------------- |
| ML framework       | PyTorch 2.3                                                               |
| Model              | Custom 2-layer Transformer classifier (128d embed, 4 heads, 256d FFN)     |
| Dataset            | IMDb Movie Reviews (or any CSV with `review` + `label` columns)           |
| Coordinator server | Flask 3.0 + Flask-CORS                                                    |
| Production WSGI    | Gunicorn (1 worker, 4 threads)                                            |
| Client networking  | Python `requests` with gzip compression + exponential backoff             |
| Real-time updates  | Server-Sent Events (SSE)                                                  |
| Credit ledger      | SQLite 3 (3 tables with foreign keys)                                     |
| Data distribution  | `pandas` + `scikit-learn` (stratified splitting)                          |
| Frontend           | Vanilla HTML/CSS/JS, Chart.js 4.4, Google Fonts (Syne, Space Mono, Inter) |
| CI                 | GitHub Actions + flake8                                                   |
| OS support         | Windows 10/11, macOS, Linux                                               |

---

## Repository Structure

```
net-neutral-ai/
├── backend/
│   ├── client/
│   │   ├── client.py              # Client entry point — full federated training lifecycle
│   │   ├── model.py               # TransformerClassifier (2-layer, 1.5M params)
│   │   ├── train.py               # Local training loop with epoch callbacks
│   │   ├── data.py                # Dataset loading, vocabulary, tokenisation, DataLoader factories
│   │   ├── requirements.txt       # Client dependencies (GPU)
│   │   └── requirements-cpu.txt   # Client dependencies (CPU-only PyTorch)
│   ├── coordinator/
│   │   ├── server.py              # Flask server — 20+ API endpoints, SSE, state machine
│   │   ├── fedavg.py              # Weighted FedAvg with validation + NaN/Inf guards
│   │   ├── evaluate.py            # Global model evaluation — accuracy, precision, recall, F1
│   │   ├── credits.py             # SQLite credit ledger — 3 tables, FK constraints, leaderboard
│   │   ├── data_distributor.py    # CSV upload → 80/20 split → stratified client sharding
│   │   ├── lan_scan.py            # LAN subnet scanner for coordinator discovery
│   │   ├── pretrain.py            # One-time baseline training script
│   │   ├── checkpoint.pt          # Pre-trained baseline model weights (committed)
│   │   ├── checkpoint_pretrained_backup.pt  # Immutable backup for session resets
│   │   ├── pretrain_log.json      # Training metadata from baseline run
│   │   ├── vocab.json             # Global vocabulary (coordinator-side, rebuilt per upload)
│   │   ├── requirements.txt       # Coordinator dependencies (local)
│   │   └── requirements-render.txt # Coordinator dependencies (Render/cloud)
│   ├── data/
│   │   └── vocab.json             # Shared vocabulary — 10,000 tokens
│   ├── shared/
│   │   ├── config.py              # Network, training, timeout, and data config
│   │   └── ip_utils.py            # LAN IP + hostname discovery utilities
│   └── demo/
│       └── run_demo.bat           # Windows Terminal launcher — 4 tiled terminals
├── frontend/
│   ├── index.html                 # 13-screen SPA — all UI screens
│   ├── styles.css                 # Dark-theme mission-control CSS (~42KB)
│   ├── app.js                     # Frontend logic — SSE, API calls, Chart.js, state machine (~78KB)
│   └── assets/                    # Status visualisation images (6 AI-generated PNGs)
├── local_data/                    # Client-side cached data shards (gitignored at runtime)
├── e2e_test.py                    # End-to-end test — server + client + full training run
├── integration_test.py            # Integration tests — imports, config, data distribution
├── WORKING.md                     # Detailed architecture explanation
├── FUTURE_SCOPES.md               # Planned improvements and roadmap
├── RENDER_DEPLOYMENT.md           # Step-by-step Render cloud deployment guide
├── Pretrained_Model.ipynb         # Jupyter notebook for baseline model exploration
├── .github/workflows/lint.yml     # GitHub Actions — flake8 lint on push/PR
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## Setup & Installation (Local Setup)

### Prerequisites

- Python 3.12+
- pip (package manager)
- Windows 10/11, macOS, or Linux

### Step 1 — Clone the repository

```bash
git clone https://github.com/Janhvesh-Patil/Net-Neutral-AI.git
cd Net-Neutral-AI
```

### Step 2 — Create virtual environment

```bash
python -m venv venv

# Windows (Command Prompt):
venv\Scripts\activate

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
# On the coordinator machine:
pip install -r backend/coordinator/requirements.txt

# On client machines (with GPU):
pip install -r backend/client/requirements.txt

# On client machines (CPU-only — smaller download):
pip install -r backend/client/requirements-cpu.txt
```

### Step 4 — Start the coordinator

Run the following command on the coordinator machine:

```bash
python backend/coordinator/server.py
```

The coordinator starts on `http://localhost:5000`. Open this URL in a browser to access the dashboard.

### Step 5 — Upload a dataset via the dashboard

1. Open the coordinator dashboard (`http://localhost:5000`) in your browser.
2. Select **+ New Job** from the top bar.
3. Drag and drop or upload a `.csv` dataset (e.g., IMDb reviews containing `review` and `label` columns).
4. (Optional) Provide a starting `.pt` checkpoint.
5. Set the job name, number of training rounds, and local epochs.
6. Click **Launch Training Session**.

> The coordinator splits the dataset, builds the global vocabulary (`vocab.json`), and shards the data for each client.

### Step 6 — Connect Client Nodes

Each client machine (or separate terminal windows on a single machine) can connect to the coordinator.

**If running on the same machine:**
Open a new terminal window for each client, activate the virtual environment, and run:

```bash
python backend/client/client.py --client_id client_A
```

_(You can use `client_A`, `client_B`, `client_C` for different nodes)._

**If running on separate machines on the same LAN:**

1. Determine the coordinator machine's local IP address (e.g., `192.168.1.50`).
2. Run the client script pointing to the coordinator's address:

```bash
python backend/client/client.py --client_id client_A --coordinator_url http://192.168.1.50:5000
```

Once registered, clients will automatically download their data shard, sync configuration, pull global models, train locally, and submit weights at each round.

---

## Running the Demo (Local LAN)

### Option A — Automated (Windows Terminal)

```bash
backend\demo\run_demo.bat
```

Opens 4 tiled terminals: coordinator + 3 clients. The coordinator starts first; clients follow after an 8-second delay.

### Option B — Manual (4 separate terminals)

**Terminal 1 — Coordinator:**

```bash
venv\Scripts\activate
python backend\coordinator\server.py
```

**Terminal 2 — Client A:**

```bash
venv\Scripts\activate
python backend\client\client.py --client_id client_A
```

**Terminal 3 — Client B:**

```bash
venv\Scripts\activate
python backend\client\client.py --client_id client_B
```

**Terminal 4 — Client C:**

```bash
venv\Scripts\activate
python backend\client\client.py --client_id client_C
```

Then open `http://localhost:5000` in a browser, upload a dataset, and click **Launch Training Session**.

---

## Database Schema

<!-- Remove this Database schema section and create a file seperately by the names schema.sql and seed.sql -->

The credit ledger uses SQLite with 3 tables, foreign key constraints, and indexes. It is auto-created at `backend/coordinator/database.db` on startup.

```sql
-- 1. Persistent client identity
CREATE TABLE clients (
    id            INTEGER  PRIMARY KEY AUTOINCREMENT,
    client_id     TEXT     NOT NULL UNIQUE,
    ip_address    TEXT     DEFAULT 'unknown',
    registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen     DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active     BOOLEAN  DEFAULT 1
);

-- 2. One row per completed round
CREATE TABLE rounds (
    id                 INTEGER  PRIMARY KEY AUTOINCREMENT,
    round_number       INTEGER  NOT NULL UNIQUE,
    started_at         DATETIME NOT NULL,
    completed_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    clients_submitted  INTEGER  NOT NULL DEFAULT 0,
    global_accuracy    REAL     NOT NULL DEFAULT 0.0,
    accuracy_delta     REAL     DEFAULT 0.0
);

-- 3. One row per client per round (FK-protected)
CREATE TABLE credits (
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
```

**Useful queries:**

```sql
-- Leaderboard
SELECT client_id, SUM(points_earned) AS total_points
FROM credits GROUP BY client_id ORDER BY total_points DESC;

-- Accuracy history across rounds
SELECT round_number, ROUND(global_accuracy * 100, 2) AS accuracy_pct
FROM rounds ORDER BY round_number ASC;

-- Client participation history
SELECT c.round, c.samples_trained, c.points_earned, r.global_accuracy
FROM credits c LEFT JOIN rounds r ON c.round = r.round_number
WHERE c.client_id = 'client_A' ORDER BY c.round;
```

**Reset between runs:**

```python
from backend.coordinator.credits import reset_db
reset_db()
```

## Testing

### Integration Tests

```bash
python integration_test.py
```

Runs 6 tests: IP discovery, configuration validation, data distribution, file structure, module imports, and syntax checks.

### End-to-End Test

```bash
python e2e_test.py
```

Spins up the coordinator, creates a dummy dataset, registers a client, runs a full 2-round training session, and verifies results.

### Per-Module Sanity Checks

Each backend module has a built-in `if __name__ == "__main__"` sanity check:

```bash
python backend/client/model.py        # Model build + forward pass
python backend/client/train.py data   # 1-epoch training round
python backend/client/data.py data    # Data loading + vocabulary
python backend/coordinator/fedavg.py  # Weighted FedAvg (7 test cases)
python backend/coordinator/evaluate.py data  # Evaluation metrics
python backend/coordinator/credits.py # Database schema + CRUD (10 test cases)
python backend/coordinator/data_distributor.py  # Dataset division (8 test cases)
python backend/shared/ip_utils.py     # IP discovery
```

---

## How It Works

For a detailed technical explanation, see [`WORKING.md`](WORKING.md). Key points:

- **The coordinator never trains.** It only distributes data, aggregates weights (FedAvg), evaluates the global model, and manages the state machine.
- **Clients do all the training.** They download a data shard once, then train locally with PyTorch for each round.
- **Privacy is architectural.** Raw text data is downloaded to the client once and never re-uploaded. Only model weight tensors travel the network.
- **Session cleanup** automatically frees memory on the coordinator after model download.
- **Checkpoint restoration** ensures every new session starts from a clean pretrained baseline — even under WSGI servers.

---

## Future Roadmap

| Version            | What Gets Added                                                      |
| ------------------ | -------------------------------------------------------------------- |
| v3 — Supabase Auth | OAuth (Google/GitHub), persistent user accounts, Supabase PostgreSQL |
| v4 — Privacy Layer | Differential privacy on gradients, secure aggregation                |
| v5 — Scale Test    | 10+ real volunteer nodes, async training rounds                      |
| v6 — Open Platform | Public client installer, community governance, multi-model support   |

See [`FUTURE_SCOPES.md`](FUTURE_SCOPES.md) for detailed plans.

---

## Contributors

| Name             | Role                                  | Key Modules                                                                      |
| ---------------- | ------------------------------------- | -------------------------------------------------------------------------------- |
| Janhvesh Patil   | ML Pipeline & Frontend                | model.py, data.py, train.py, evaluate.py, pretrain.py, fedavg.py, frontend UI/UX |
| Tejas Kolekar    | Coordinator Server & Database         | server.py, credits.py, data_distributor.py, database schema                      |
| Atharv Huilgol   | Client App, Networking & Project Lead | client.py, network layer, config.py, ip_utils.py                                 |
| Bhoomika Salunke | Client App & Networking               | client.py, network layer, integration testing                                    |

---

## Acknowledgements

- IMDb dataset: [Maas et al., 2011](http://www.aclweb.org/anthology/P11-1015)
- Federated Learning: [McMahan et al., 2017](https://arxiv.org/abs/1602.05629) — _Communication-Efficient Learning of Deep Networks from Decentralized Data_
- Built for GitHub DevDays Hackathon 2026

---

_Net-Neutral AI — GitHub DevDays Hackathon 2026_
