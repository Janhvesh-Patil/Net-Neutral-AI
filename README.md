# Net-Neutral AI

> *Democratising AI training — one idle GPU at a time.*

[![Lint](https://github.com/Janhvesh-Patil/net-neutral-ai/actions/workflows/lint.yml/badge.svg)](https://github.com/Janhvesh-Patil/net-neutral-ai/actions/workflows/lint.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What is Net-Neutral AI?

Training a frontier AI model costs upwards of $100M and is accessible to fewer than five organisations globally. Meanwhile, billions of devices — student laptops, gaming PCs, smartphones — sit idle with capable GPUs doing nothing.

**Net-Neutral AI is a federated learning platform** that recruits these idle devices as volunteer compute nodes, enabling collaborative AI model training without any single company controlling the infrastructure.

**Two guarantees by design:**
- **Zero Raw Data Leakage**: Raw data never leaves a device — only learned weight updates travel across the network.
- **Credit-Based Attribution**: Every contributor is tracked and rewarded transparently through a persistent credit ledger.

This is the net neutrality principle applied to AI compute: equal access, distributed power, no gatekeepers.

---

## System Architecture

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

### One round of federated training:
1. **Model Broadcast**: Coordinator sends global model to all registered clients.
2. **Local Training**: Each client trains locally on their designated data shard.
3. **Weight Upload**: Clients send back updated weights (compressed via GZIP) — not data.
4. **Aggregation (FedAvg)**: Coordinator runs Federated Averaging (FedAvg), averaging all weight updates.
5. **Evaluation**: Coordinator evaluates new global model on a held-out validation set.
6. **Repeat**: Repeat for a configurable number of rounds (e.g. 5 rounds).

---

## Web Frontend Dashboard Features

The platform includes a real-time web dashboard that streamlines configuration, monitoring, and model exports:

- **Glassmorphic Dark UI**: Built with modern typography, smooth gradients, and rich aesthetics.
- **Dynamic Session Settings**: Configure expected client counts, training rounds, and local training epochs dynamically before starting.
- **SSE Real-Time Sync**: Fully synchronized event stream via Server-Sent Events (SSE). Status changes, client joins, and logging metrics update instantly without page refreshes.
- **Network Topology Visualizer**: An SVG canvas showing live connections and animation states (idle, downloading, training, uploading) for all client nodes.
- **Tabbed Round Epoch Metrics**: Tabular summaries of epoch loss and accuracy for each round to monitor training convergence.
- **Persistent Timeline & Credits**: Rebuilds transaction logs and credits ledger directly from the SQLite database.
- **Global Model Download**: Export the final trained `checkpoint.pt` directly from the web browser after execution is complete.

---

## Tech Stack

| Component | Technology |
|---|---|
| **ML Framework** | PyTorch 2.3 |
| **Model** | Custom 2-layer Transformer classifier |
| **Coordinator Server** | Flask 3.0 (with Gunicorn WSGI for production) |
| **Client Networking** | Python `requests` (with connection retries and GZIP uploads) |
| **Database** | SQLite 3 |
| **Web Interface** | HTML5 / JavaScript (ES6) / CSS3 (Vanilla Glassmorphism) |
| **Real-time Pipeline** | Server-Sent Events (SSE) stream |
| **Testing** | Automated Python integration and end-to-end tests |

---

## Repository Structure

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
│   │   ├── checkpoint.pt           # Current global model weights
│   │   ├── checkpoint_pretrained_backup.pt  # Initial baseline pre-trained checkpoint
│   │   ├── pretrain_log.json       # Pretraining metrics & metadata log
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
├── local_data/                     # Directory dynamically generated to hold client shards
├── integration_test.py             # Automated network and data integration tests
├── e2e_test.py                     # Fully simulated end-to-end local training test
└── README.md
```

---

## Setup & Deployment

### Prerequisites
- Python 3.12
- Windows 10/11 (with Windows Terminal for automated local demo)
- All local machines connected to the same WiFi network (for local runs)

---

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Janhvesh-Patil/Net-Neutral-AI.git -b testing_site
cd Net-Neutral-AI
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install Dependencies
```bash
# On coordinator machine:
pip install -r backend/coordinator/requirements.txt

# On client machines:
pip install -r backend/client/requirements.txt
```

---

### Step 4 — Launch & Upload Dataset via Web Dashboard

1. **Start Coordinator Server**:
   ```bash
   python backend/coordinator/server.py
   ```
2. **Open Web Browser**: Navigate to `http://localhost:5000` (or `http://[COORDINATOR_IP]:5000`).
3. **Configure & Start Setup**: Enter the number of expected clients.
4. **Upload CSV Dataset**: On the setup dashboard, upload a `.csv` dataset file.
   * **Dataset Constraints**: The CSV must strictly contain a `review` (text) and `label` (binary `0` or `1`) column.
   * **Processing Flow**: The coordinator automatically partitions the uploaded file (80% training distributed dynamically to clients, 20% validation securely retained on the server) and generates the vocabulary `vocab.json` file.

---

### Step 5 — Configure Network

#### Option A: Automatic CLI Arguments (Recommended)
You can directly pass the coordinator URL when running client scripts without modifying configuration files:
```bash
python backend/client/client.py --client_id client_A --coordinator_url http://[COORDINATOR_IP]:5000
```

#### Option B: Configuration File (Optional)
Edit `backend/shared/config.py` on client machines:
```python
COORDINATOR_IP = "192.168.1.X"   # Replace with coordinator machine's local IPv4 IP
CLIENT_ID      = "client_A"       # Set unique client ID: client_A, client_B, etc.
```

---

### Step 6 — Verify Pre-Trained Checkpoint
Verify the validity and accuracy of the baseline pre-trained checkpoint:
```bash
python backend/coordinator/pretrain.py backend/data --verify-only
```
Expected output: `✓ Checkpoint is demo-ready.` (Accuracy ~80.95%).

---

## Running the Demo

### Option A — Automated (Windows Terminal)
Run the script from the root directory or double-click it:
```bash
backend\demo\run_demo.bat
```
Opens 4 tiled terminals simultaneously: one coordinator and three client nodes. Training starts automatically once clients connect and setup is finalized on the frontend.

### Option B — Manual (4 Separate Terminals)

* **Terminal 1 — Coordinator:**
  ```bash
  python backend/coordinator/server.py
  ```
* **Terminal 2 — Client A:**
  ```bash
  python backend/client/client.py --client_id client_A
  ```
* **Terminal 3 — Client B:**
  ```bash
  python backend/client/client.py --client_id client_B
  ```
* **Terminal 4 — Client C:**
  ```bash
  python backend/client/client.py --client_id client_C
  ```

---

## Cloud Deployment (Render.com)

The coordinator is fully optimized for cloud deployment:
* **WSGI Production Configuration**: Run with Gunicorn using `backend/coordinator/requirements-render.txt`.
* **Render Web Service Settings**:
  * **Build Command**: `pip install -r backend/coordinator/requirements-render.txt`
  * **Start Command**: `gunicorn -w 1 --threads 4 -b 0.0.0.0:$PORT --timeout 300 backend.coordinator.server:app`
* **Automated Memory Cleanup**: Employs an aggressive memory cleanup cycle following model downloads (removing uploaded datasets, temporary client weight files, and in-memory caches) to respect Render's 512MB limits.

---

## Database Schema (SQLite)

Credits are persisted in `backend/coordinator/database.db` automatically at server startup:

```sql
-- One row per client submission per round
CREATE TABLE credits (
    id               INTEGER  PRIMARY KEY AUTOINCREMENT,
    client_id        TEXT     NOT NULL,
    round            INTEGER  NOT NULL,
    samples_trained  INTEGER  NOT NULL DEFAULT 0,
    time_seconds     REAL     NOT NULL DEFAULT 0.0,
    points_earned    INTEGER  NOT NULL DEFAULT 0,
    timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- One row per completed round
CREATE TABLE rounds (
    id                 INTEGER  PRIMARY KEY AUTOINCREMENT,
    round_number       INTEGER  NOT NULL UNIQUE,
    started_at         DATETIME NOT NULL,
    completed_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    clients_submitted  INTEGER  NOT NULL DEFAULT 0,
    global_accuracy    REAL     NOT NULL DEFAULT 0.0,
    accuracy_delta     REAL     DEFAULT 0.0
);
```

---

## Testing & Verification

Ensure code correctness and network integrity by running the test suite:

### 1. Integration Tests
Verifies local IP discovery, core configurations, and file structures:
```bash
python integration_test.py
```

### 2. End-to-End Simulation
Simulates full dataset upload, client registration, and local training cycles automatically:
```bash
python e2e_test.py
```

---

## Contributors

| Name             | Role                                  | Modules |
|------------------|---------------------------------------|---|
| Janhvesh Patil   | ML Pipeline                           | model.py, data.py, train.py, evaluate.py, pretrain.py, fedavg.py |
| Tejas Kolekar    | Coordinator Server & Database         | server.py, credits.py, database |
| Atharv Huilgol   | Technical Architect                   | client.py, network layer |
| Bhoomika Salunke | Client App & Networking               | client.py, network layer |

---

## Acknowledgements

- Federated Learning: [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)

---

*Net-Neutral AI — GitHub DevDays Hackathon 2026*
