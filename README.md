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
- Raw data never leaves a device — only learned weight updates travel across the network
- Every contributor is tracked and rewarded transparently through a persistent credit ledger

This is the net neutrality principle applied to AI compute: equal access, distributed power, no gatekeepers.

---

## System Architecture

```
                    ┌─────────────────────────┐
                    │   Coordinator (Flask)   │
                    │                         │
                    │  ┌─────────┐ ┌────────┐ │
                    │  │ FedAvg  │ │Credits │ │
                    │  │ Engine  │ │ SQLite │ │
                    │  └─────────┘ └────────┘ │
                    └────────────┬────────────┘
                                 │ Local WiFi
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼──────┐ ┌─────────▼──────┐ ┌────────▼───────┐
    │   Client A     │ │   Client B     │ │   Client C     │
    │                │ │                │ │                │
    │  Local data    │ │  Local data    │ │  Local data    │
    │  stays here    │ │  stays here    │ │  stays here    │
    └────────────────┘ └────────────────┘ └────────────────┘

    ↑ Weight updates (what model learned) travel UP to coordinator
    ↓ Global model (averaged weights)     travel DOWN to clients
    ✗ Raw data never leaves any device
```

**One round of federated training:**
1. Coordinator sends global model to all clients
2. Each client trains locally on their own data shard
3. Clients send back updated weights — not data
4. Coordinator runs FedAvg — averages all weight updates
5. Coordinator evaluates new global model on held-out validation set
6. Repeat for 5 rounds

---

## Results

Baseline checkpoint — pre-trained on 15,000 samples for 5 epochs on GPU:

| Metric | Value |
|---|---|
| Best validation accuracy | **80.95%** |
| Correct / Total | 1,619 / 2,000 |
| Eval loss | 0.5078 |
| Training time | 40.0 seconds |
| Model parameters | 1,561,602 |
| Dataset | IMDb Movie Reviews (15K training subset) |

**Per-class breakdown (baseline checkpoint):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.8194 | 0.7870 | 0.8029 | 986 |
| Positive | 0.8006 | 0.8314 | 0.8157 | 1,014 |
| **Macro avg** | **0.8100** | **0.8092** | **0.8093** | 2,000 |

**Pretraining accuracy progression:**

| Epoch | Val Accuracy | Delta |
|---|---|---|
| 1 | 76.35% | baseline |
| 2 | 79.25% | +2.90% |
| 3 | 80.90% | +1.65% |
| 4 | 80.85% | −0.05% |
| 5 | **80.95%** | +0.10% |

*Federated round results will be updated after demo recording.*
*See `coordinator/pretrain_log.json` for full training metadata.*

---

## Tech Stack

| Component | Technology |
|---|---|
| ML framework | PyTorch 2.3 |
| Model | Custom 2-layer Transformer classifier |
| Dataset | IMDb Movie Reviews (HuggingFace) |
| Coordinator server | Flask 3.0 |
| Client networking | Python requests |
| Credit ledger | SQLite 3 (built-in) |
| CI | GitHub Actions + flake8 |
| OS | Windows 10/11 |

---

## Repository Structure

```
net-neutral-ai/
├── client/
│   ├── client.py         # Client entry point — runs federated training loop
│   ├── model.py          # TransformerClassifier architecture
│   ├── train.py          # Local training loop (one round)
│   ├── data.py           # Dataset loading, tokenisation, DataLoader factory
│   └── requirements.txt
├── coordinator/
│   ├── server.py         # Flask coordinator — all API endpoints
│   ├── fedavg.py         # Weighted FedAvg algorithm
│   ├── evaluate.py       # Model evaluation — accuracy, precision, recall, F1
│   ├── credits.py        # SQLite credit ledger — read/write operations
│   ├── pretrain.py       # One-time baseline training script
│   ├── checkpoint.pt     # Pre-trained baseline model weights (committed)
│   ├── pretrain_log.json # Training metadata from baseline run
│   └── requirements.txt
├── data/
│   ├── imdb_train.csv    # 40,000 training reviews (download separately)
│   ├── imdb_test.csv     # 10,000 test reviews (download separately)
│   └── vocab.json        # Shared vocabulary — 10,000 tokens (committed)
├── shared/
│   └── config.py         # Coordinator IP, round count, hyperparameters
├── demo/
│   └── run_demo.bat      # Windows Terminal launcher — 4 tiled terminals
├── .github/
│   └── workflows/
│       └── lint.yml      # GitHub Actions — flake8 lint on push
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.12
- All machines on the same WiFi network
- Windows 10/11
- Windows Terminal (for tiled demo launcher)

### Step 1 — Clone on all machines

```bash
git clone https://github.com/Janhvesh-Patil/net-neutral-ai.git
cd net-neutral-ai
```

### Step 2 — Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
# On coordinator machine:
pip install -r coordinator/requirements.txt

# On client machines:
pip install -r client/requirements.txt
```

### Step 4 — Download dataset

Download IMDb dataset from [HuggingFace](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews) and place files in `data/`:
- `data/imdb_train.csv` (40,000 rows)
- `data/imdb_test.csv` (10,000 rows)

`vocab.json` and `checkpoint.pt` are already committed — no need to regenerate them.

### Step 5 — Configure network

Find the coordinator machine's local IP:
```bash
ipconfig   # look for IPv4 Address under the WiFi adapter
```

Edit `shared/config.py` on every machine:
```python
COORDINATOR_IP = "192.168.1.X"   # replace with coordinator machine's IP
CLIENT_ID      = "client_A"       # set per machine: client_A, client_B, client_C
```

### Step 6 — Verify checkpoint

```bash
python coordinator/pretrain.py data --verify-only
```

Expected output: `✓ Checkpoint is demo-ready.` with accuracy ~80.95%.

### Step 7 — Test network connectivity

From any client machine, open a browser and go to:
```
http://[COORDINATOR_IP]:5000/status
```
Should return JSON. If not — add a Windows Firewall exception for port 5000 on the coordinator machine.

---

## Running the Demo

### Option A — Automated (Windows Terminal)

```bash
demo\run_demo.bat
```

Opens 4 tiled terminals simultaneously. Coordinator starts first, clients follow after a short delay.

### Option B — Manual (4 separate terminals)

**Terminal 1 — Coordinator:**
```bash
venv\Scripts\activate
python coordinator\server.py
```

**Terminal 2 — Client A:**
```bash
venv\Scripts\activate
python client\client.py --client_id client_A
```

**Terminal 3 — Client B:**
```bash
venv\Scripts\activate
python client\client.py --client_id client_B
```

**Terminal 4 — Client C:**
```bash
venv\Scripts\activate
python client\client.py --client_id client_C
```

Wait for all 3 clients to register. Training starts automatically once all clients are connected.

---

## Database

The credit ledger uses SQLite — no installation or configuration required. It is auto-created at `coordinator/database.db` when the coordinator starts.

**Schema:**

```sql
-- One row per client per round
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

**Useful queries:**

```sql
-- Leaderboard
SELECT client_id, SUM(points_earned) AS total_points
FROM credits GROUP BY client_id ORDER BY total_points DESC;

-- Accuracy history across rounds
SELECT round_number, ROUND(global_accuracy * 100, 2) AS accuracy_pct
FROM rounds ORDER BY round_number ASC;
```

**Export results to CSV after demo:**
```bash
sqlite3 coordinator/database.db ".headers on" ".mode csv" ".output results.csv" "SELECT * FROM rounds;" ".quit"
```

**Reset between runs:**
```python
from coordinator.credits import reset_db
reset_db()
```

---

## Pre-Demo Checklist

Run through this before every recording session:

- [ ] All machines connected to same WiFi
- [ ] `config.py` has correct `COORDINATOR_IP` on all machines
- [ ] `coordinator/checkpoint.pt` exists — verify with `--verify-only`
- [ ] Virtual environment activated on all machines
- [ ] Dataset files present in `data/` on all client machines
- [ ] Port 5000 reachable — test from browser on client machine
- [ ] Windows Terminal installed for tiled view
- [ ] Recording software ready (OBS or Xbox Game Bar)
- [ ] Full dry run completed — all 5 rounds, no crashes
- [ ] `database.db` reset before actual recording run

---

## Future Roadmap

| Version | What Gets Added |
|---|---|
| v2 — Internet-Ready | Cloud coordinator, internet node communication |
| v3 — Privacy Layer | Differential privacy on gradients, secure aggregation |
| v4 — Scale Test | 10+ real volunteer nodes, async training rounds |
| v5 — Open Platform | Public client installer, community governance |

---

## Contributors

| Name             | Role                                  | Modules |
|------------------|---------------------------------------|---|
| Janhvesh Patil   | ML Pipeline                           | model.py, data.py, train.py, evaluate.py, pretrain.py, fedavg.py |
| Tejas Kolekar    | Coordinator Server & Database         | server.py, credits.py, database |
| Atharv Huilgol   | Client App, Networking & Project Lead | client.py, network layer |
| Bhoomika Salunke | Client App & Networking               | client.py, network layer |

---

## Acknowledgements

- IMDb dataset: [Maas et al., 2011](http://www.aclweb.org/anthology/P11-1015)
- Federated Learning: [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)
- Built for GitHub DevDays Hackathon 2026

---

*Net-Neutral AI — GitHub DevDays Hackathon 2026*