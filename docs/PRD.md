# Product Requirements Document (PRD)
# Net-Neutral AI — Federated Learning Platform

**Version:** 1.0  
**Date:** 2026-07-28  
**Authors:** Janhvesh Patil, Tejas Kolekar, Atharv Huilgol, Bhoomika Salunke  
**Status:** Active Development (Prototype)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Target Users & Personas](#4-target-users--personas)
5. [Core Principles & Design Philosophy](#5-core-principles--design-philosophy)
6. [Product Features & Requirements](#6-product-features--requirements)
7. [User Stories & Flows](#7-user-stories--flows)
8. [Data Requirements](#8-data-requirements)
9. [Dashboard & UI Requirements](#9-dashboard--ui-requirements)
10. [Performance & Scalability Requirements](#10-performance--scalability-requirements)
11. [Security & Privacy Requirements](#11-security--privacy-requirements)
12. [Deployment Requirements](#12-deployment-requirements)
13. [Testing Requirements](#13-testing-requirements)
14. [Future Scope & Roadmap](#14-future-scope--roadmap)
15. [Success Metrics](#15-success-metrics)

---

## 1. Executive Summary

**Net-Neutral AI** is an open-source federated learning platform designed to democratise AI model training by pooling idle compute resources from volunteer devices (student laptops, gaming PCs, etc.). The platform enables collaborative, privacy-preserving AI model training without any single company controlling the infrastructure.

The core product consists of:
- A **Coordinator Server** that orchestrates federated training sessions.
- A **Client Agent** that runs on volunteer compute nodes and performs local model training.
- A **Web Dashboard** providing real-time monitoring, topology visualisation, and model export.

The guiding philosophy is the application of the *net neutrality principle* to AI compute: **equal access, distributed power, no gatekeepers**.

---

## 2. Problem Statement

### The Core Problem

Training frontier AI models is prohibitively expensive — often exceeding $100M — and is accessible to fewer than five organisations globally. At the same time, billions of devices worldwide sit idle with capable GPUs performing no useful computation.

This creates two compounding issues:
1. **Compute concentration**: AI innovation is gated behind massive capital investment, limiting research access.
2. **Resource waste**: Enormous amounts of potentially useful compute sit unused on commodity devices.

### The Privacy Challenge

Traditional distributed training approaches require sharing raw data with a central server. This is a significant barrier for participants who:
- Cannot legally share data (healthcare, finance, legal domains).
- Do not wish to relinquish ownership of their data.
- Operate under GDPR, HIPAA, or similar data regulations.

### The Attribution Challenge

There is no existing transparent mechanism to reward volunteer compute contributors, creating a free-rider problem that discourages sustained participation.

---

## 3. Goals & Non-Goals

### Goals

| Goal | Priority |
|------|----------|
| Enable collaborative AI model training across geographically distributed volunteer devices | P0 |
| Guarantee that raw training data never leaves any client device | P0 |
| Provide a real-time web dashboard for training monitoring and management | P0 |
| Implement a transparent credit/attribution system for contributor reward | P1 |
| Support both local (LAN) and cloud-hosted coordinator deployments | P1 |
| Allow dynamic configuration of training parameters (rounds, epochs, batch size) | P1 |
| Enable one-click export of the final trained global model | P1 |
| Provide automated integration and end-to-end test suites | P2 |
| Support automated demo setup using Windows Terminal | P2 |

### Non-Goals

- **This is not a general-purpose ML training framework** — the initial prototype targets binary text classification only.
- **This is not a blockchain system** — credits are tracked in SQLite, not a distributed ledger.
- **This is not a real-time collaborative editor** — clients train asynchronously, not simultaneously.
- **This is not a data marketplace** — the platform does not facilitate data buying/selling.
- **This does not guarantee differential privacy** — formal DP mechanisms (noise injection) are not implemented in v1.
- **This does not provide model serving/inference** — the output is a `.pt` checkpoint file; serving is out of scope.

---

## 4. Target Users & Personas

### Persona 1: The Coordinator (Research Lead / ML Engineer)

**Background:** A researcher or ML engineer who owns a dataset and needs to train a model without centralising raw data.

**Needs:**
- Upload a dataset securely and have it automatically sharded.
- Configure training hyperparameters (rounds, epochs) through a web UI.
- Monitor training progress in real-time via a dashboard.
- Download the final trained model once training is complete.

**Pain Points:**
- Cannot afford cloud GPU compute for large-scale training.
- Cannot share raw data with a centralised service due to compliance.

---

### Persona 2: The Client (Volunteer Compute Provider)

**Background:** A student, developer, or enthusiast with a gaming PC or laptop with idle GPU capacity.

**Needs:**
- Join a federated training session with a single CLI command.
- Be assured their local data never leaves their machine.
- Be recognised and rewarded for their compute contribution.
- Understand their training progress via a live dashboard view.

**Pain Points:**
- Has spare GPU capacity but no way to monetise or contribute it.
- Concerned about privacy and data security.

---

### Persona 3: The Observer (Developer / Stakeholder)

**Background:** A developer, investor, or academic evaluating the platform.

**Needs:**
- View the network topology and understand how the system works.
- See training metrics, leaderboard data, and accuracy evolution.
- Run demo mode to see the platform in action without a full multi-machine setup.

---

## 5. Core Principles & Design Philosophy

### Principle 1: Privacy by Architecture
Raw data is **architecturally prevented** from leaving a device. Only learned weight updates (model parameters) travel across the network. This is not a policy constraint; it is enforced by the design of the client agent.

### Principle 2: Transparent Attribution
Every training contribution is recorded in a persistent SQLite credit ledger. The credit formula is deterministic: **1 point per 5 samples trained**. The leaderboard is public and queryable.

### Principle 3: Simplicity of Participation
Joining the network requires only:
1. `git clone`
2. `pip install -r requirements.txt`
3. `python client.py --client_id <id> --coordinator_url <url>`

No complex key management, wallet setup, or registration process.

### Principle 4: Real-Time Transparency
All training state — client connections, round progress, accuracy metrics, epoch loss — is streamed in real-time to any connected browser via Server-Sent Events (SSE), with no page refreshes required.

### Principle 5: Resilience to Partial Failures
The FedAvg aggregation algorithm is designed to tolerate late or missing client submissions. If a client's weights are corrupted or have mismatched keys, they are skipped gracefully without failing the round.

---

## 6. Product Features & Requirements

### F1 — Client Registration & Discovery

| ID | Requirement | Priority |
|----|-------------|----------|
| F1.1 | Clients must be able to register with the coordinator via a single HTTP POST request. | P0 |
| F1.2 | Client registration must persist to SQLite (client_id, IP address, registration timestamp). | P0 |
| F1.3 | Client registration must trigger a real-time SSE `client_joined` event to all connected browsers. | P1 |
| F1.4 | Clients must support automatic retry on registration failure (configurable attempts and delay). | P1 |
| F1.5 | The system must support automatic LAN IP discovery via `lan_scan.py` as an alternative to manual configuration. | P2 |

### F2 — Dataset Management

| ID | Requirement | Priority |
|----|-------------|----------|
| F2.1 | The Coordinator must accept a CSV file upload from the web dashboard. | P0 |
| F2.2 | Uploaded CSV files must contain exactly `review` (text) and `label` (binary 0/1) columns; all other formats must be rejected with a descriptive error. | P0 |
| F2.3 | The system must automatically split uploaded data 80% training / 20% validation. | P0 |
| F2.4 | Training data must be sharded equally across all registered clients. | P0 |
| F2.5 | The coordinator must build and serve a global vocabulary (`vocab.json`) derived from the full training corpus for consistent tokenisation across all clients. | P0 |
| F2.6 | Sharded data must be served to individual clients over HTTP (not broadcast to all). | P0 |
| F2.7 | A client must only ever receive its own data shard, never another client's data or the full dataset. | P0 |

### F3 — Federated Training Loop

| ID | Requirement | Priority |
|----|-------------|----------|
| F3.1 | The coordinator must support a configurable number of federated training rounds (default: 5). | P0 |
| F3.2 | Each round must: broadcast global model → wait for all clients to submit weights → run FedAvg → evaluate → advance round. | P0 |
| F3.3 | Clients must download the global model, train locally for a configurable number of epochs, and upload updated weights. | P0 |
| F3.4 | Weight uploads must be compressed using GZIP to minimise network transfer size. | P1 |
| F3.5 | The FedAvg algorithm must use sample-count weighted averaging. | P0 |
| F3.6 | FedAvg must gracefully skip clients with corrupted, NaN-containing, or key-mismatched weight files. | P1 |
| F3.7 | Clients must poll the coordinator every 5 seconds (configurable) to check for round state transitions. | P0 |
| F3.8 | The coordinator must wait up to 90 seconds (configurable) for all clients to submit before timing out a round. | P1 |

### F4 — Model Evaluation

| ID | Requirement | Priority |
|----|-------------|----------|
| F4.1 | After each FedAvg aggregation, the coordinator must evaluate the new global model on the held-out validation set. | P0 |
| F4.2 | Evaluation must report accuracy, precision, recall, F1 (per-class and macro-averaged). | P0 |
| F4.3 | Evaluation must report the delta accuracy vs. the previous round. | P1 |
| F4.4 | Evaluation must be memory-efficient (cache the validation DataLoader; free model memory after each evaluation). | P1 |

### F5 — Credit & Attribution System

| ID | Requirement | Priority |
|----|-------------|----------|
| F5.1 | Every client submission must be logged to the `credits` table in SQLite with: client_id, round, samples_trained, time_seconds, points_earned, timestamp. | P0 |
| F5.2 | Every completed round must be logged to the `rounds` table with: round_number, started_at, completed_at, clients_submitted, global_accuracy, accuracy_delta. | P0 |
| F5.3 | Points formula: `floor(samples_trained / 5)` points per submission. | P0 |
| F5.4 | The system must expose a public `/leaderboard` endpoint returning ranked totals for all clients. | P1 |
| F5.5 | The leaderboard must persist across page refreshes and reconnections. | P1 |

### F6 — Model Export

| ID | Requirement | Priority |
|----|-------------|----------|
| F6.1 | After training completes, the final `checkpoint.pt` must be downloadable from the web dashboard. | P0 |
| F6.2 | The download endpoint must trigger post-download session cleanup to respect cloud memory constraints. | P1 |
| F6.3 | After cleanup, the checkpoint must be automatically restored to the pre-trained baseline so the next session starts cleanly. | P1 |

### F7 — Per-Epoch Metrics Reporting

| ID | Requirement | Priority |
|----|-------------|----------|
| F7.1 | Clients must report per-epoch training metrics (epoch, loss, accuracy, samples, round) to the coordinator after each local epoch. | P1 |
| F7.2 | The coordinator must broadcast received epoch data as SSE `epoch_update` events to all connected browsers. | P1 |
| F7.3 | The dashboard must display epoch metrics in a tabbed view, organised per round. | P1 |

---

## 7. User Stories & Flows

### Flow 1: Coordinator Session Setup

```
1. Coordinator starts server:  python backend/coordinator/server.py
2. Opens browser → http://localhost:5000
3. Enters expected client count → clicks "Launch Coordinator Dashboard"
4. Uploads CSV dataset (review + label columns)
5. System validates dataset, builds vocabulary, shards data
6. Waits for client nodes to connect
7. Clicks "Start Training" once all clients have joined
8. Monitors real-time dashboard during training
9. After all rounds complete, downloads checkpoint.pt
```

### Flow 2: Client Node Participation

```
1. Client clones the repository
2. Installs dependencies: pip install -r backend/client/requirements.txt
3. Launches client: python backend/client/client.py --client_id client_A --coordinator_url http://[COORDINATOR_IP]:5000
4. Client registers with coordinator → appears on coordinator dashboard
5. Client downloads its assigned data shard from coordinator
6. Client downloads global model → trains locally → uploads updated weights
7. Process repeats for each federated round
8. Client sees its credits accumulate in real-time on the dashboard
```

### Flow 3: Cloud Deployment (Render)

```
1. Deploy coordinator to Render using requirements-render.txt + Gunicorn
2. Clients connect to: python client.py --coordinator_url https://your-app.onrender.com
3. All other steps identical to local flow
```

### Flow 4: Automated Demo

```
1. Run: backend\demo\run_demo.bat
2. 4 Windows Terminal panes open simultaneously
3. Pane 1: Coordinator server starts
4. Panes 2-4: Three client agents start
5. Once all 3 clients register, training begins automatically
```

---

## 8. Data Requirements

### Dataset Input Format

```
CSV file with exactly two columns:
  - review  : string (text input for classification)
  - label   : integer (binary: 0 = negative, 1 = positive)
```

### Data Partition Strategy

| Partition | Allocation | Location | Purpose |
|-----------|-----------|----------|---------|
| Training pool | 80% of uploaded data | Split equally across client devices | Local training (never centralized) |
| Validation set | 20% of uploaded data | Retained on coordinator server | Global model evaluation |

### Vocabulary

- Built from the full training corpus (up to `VOCAB_SIZE = 10,000` tokens).
- Persisted as `vocab.json` on the coordinator.
- Distributed to clients before local training begins.
- Ensures consistent tokenisation across heterogeneous client environments.

### Model Checkpoint Format

- Format: PyTorch `state_dict` serialised via `torch.save()`.
- File: `checkpoint.pt` (~6 MB for the 2-layer Transformer).
- Baseline: `checkpoint_pretrained_backup.pt` (pre-trained to ~80.95% accuracy for demo use).

---

## 9. Dashboard & UI Requirements

### Design Language
- **Glassmorphic Dark UI** with modern typography.
- Smooth gradients, subtle micro-animations, hover effects.
- Fully single-page application (no page reloads for status updates).

### Coordinator Dashboard Components

| Component | Description |
|-----------|-------------|
| **Session Configuration Panel** | Input fields for expected client count, total rounds, local epochs. |
| **Dataset Upload Widget** | Drag-and-drop or click-to-upload CSV file interface with validation feedback. |
| **Network Topology Visualiser** | SVG canvas showing live coordinator-client connections with animated states. |
| **Training Progress Bar** | Shows current round out of total rounds with percentage. |
| **Accuracy History Chart** | Chart showing global accuracy per round. |
| **Round Epoch Metrics Table** | Tabbed view — one tab per round, showing per-client epoch loss and accuracy. |
| **Credits Leaderboard** | Ranked table showing client_id, total points, total samples, rounds participated. |
| **System Log** | Scrolling real-time log of SSE `sys_log` events. |
| **Download Model Button** | Appears after training completes. Downloads `checkpoint.pt`. |

### Real-Time Synchronisation (SSE Events)

| Event Name | Trigger | Payload |
|------------|---------|---------|
| `init` | Browser connects to `/events` | Full current state |
| `client_joined` | New client registers | `{client_id, ip, total_clients}` |
| `epoch_update` | Client reports epoch metrics | `{client_id, epoch, loss, accuracy, samples, round}` |
| `accuracy_update` | Round aggregation completes | `{round, accuracy, clients_submitted}` |
| `round_start` | New round begins | `{round, status}` |
| `sys_log` | Internal coordinator event | `{message, level}` |
| `training_done` | All rounds complete | `{final_accuracy, total_rounds}` |
| `session_cleanup` | Post-download cleanup | `{files_removed}` |

---

## 10. Performance & Scalability Requirements

| Requirement | Target |
|-------------|--------|
| Maximum concurrent clients (prototype) | 3 |
| Coordinator server response latency | < 500ms for all API endpoints |
| SSE keepalive interval | 20 seconds |
| Client poll interval | 5 seconds |
| Client registration retry | 3 attempts, 10-second delay |
| Round submission timeout | 90 seconds |
| Client data wait timeout | 300 seconds |
| Memory budget (Render free tier) | < 512 MB peak |
| Session cleanup trigger | Immediately after model download |

---

## 11. Security & Privacy Requirements

| Requirement | Implementation |
|-------------|----------------|
| Raw data never leaves client | Architecturally enforced — only weight `.pt` files are transmitted |
| Filename sanitisation on upload | Server rejects filenames containing `..`, `/`, `\` characters |
| File type validation | Coordinator only accepts `.csv` uploads |
| Weight file validation | FedAvg validates all tensors for NaN, Inf, and key mismatches before aggregation |
| Thread safety | All global state mutations protected by `threading.RLock()` |
| CORS | Enabled globally via `flask_cors` for cross-origin browser access |
| No authentication (v1) | Single-session trust model; multi-tenant auth is a future scope item |
| SQLite journal mode | `DELETE` mode (not WAL) to prevent database corruption on ephemeral storage |

---

## 12. Deployment Requirements

### Local Deployment

| Requirement | Value |
|-------------|-------|
| OS | Windows 10/11 (demo automation); Linux/macOS compatible for server/clients |
| Python version | 3.12 |
| Coordinator port | 5000 (configurable via `COORDINATOR_PORT` env var) |
| Network | All devices on same WiFi for local mode |
| ML framework | PyTorch 2.3 |
| Server | Flask 3.0 (development) |

### Cloud Deployment (Render.com)

| Requirement | Value |
|-------------|-------|
| WSGI server | Gunicorn |
| Worker configuration | `-w 1 --threads 4` (single process, multi-thread) |
| Request timeout | 300 seconds (model upload/download) |
| Memory limit | 512 MB (free tier) |
| Storage | Ephemeral (SQLite + checkpoints reset on restart) |
| Environment variables | `PORT` (auto-set by Render) |

---

## 13. Testing Requirements

### Integration Tests (`integration_test.py`)
- Validate local IP discovery functions.
- Verify core configuration values.
- Assert required file structures are present.

### End-to-End Tests (`e2e_test.py`)
- Simulate full dataset upload to coordinator.
- Simulate client registration flow.
- Simulate local training cycles including data shard download.
- Verify weight submission and FedAvg aggregation.
- Assert that global accuracy is computed and stored correctly.

### Manual Verification
- Verify pre-trained checkpoint: `python backend/coordinator/pretrain.py backend/data --verify-only`
- Expected output: accuracy ~80.95%.
- Run demo mode to confirm 4-terminal automated setup works end-to-end.

---

## 14. Future Scope & Roadmap

### Phase 2 — Local Coordinator Mode
Move heavy FedAvg and evaluation off cloud servers to the coordinator's local machine. Reduce cloud service to a lightweight HTTP relay/signalling server only.

### Phase 3 — Frontend & UX Overhaul
- Replace SSE polling with full WebSocket bi-directional communication.
- Migrate frontend to React/Tailwind for a production-quality UI.
- Surface rich per-client terminal data in the browser.

### Phase 4 — Differential Privacy
Implement formal DP noise injection on client weight updates before upload.

### Phase 5 — Multi-Model Support
Extend beyond binary text classification. Support image classification, tabular data, and regression tasks.

### Phase 6 — Authentication & Multi-Session
Add API key or OAuth-based authentication for multi-tenant environments.

### Phase 7 — Decentralised Ledger
Replace SQLite credit ledger with a smart-contract-based on-chain attribution system.

---

## 15. Success Metrics

| Metric | Target (Prototype) |
|--------|-------------------|
| Training sessions that complete all rounds without manual intervention | >= 95% |
| Baseline pre-trained accuracy (verification) | ~80.95% |
| Accuracy improvement after 5 federated rounds | > 0% vs baseline |
| Client registration → first round start latency | < 30 seconds |
| Dashboard SSE event delivery latency | < 2 seconds |
| Raw data transmitted from any client to coordinator | 0 bytes |
| Test suite pass rate | 100% |

---

*This document is maintained by the Net-Neutral AI team. For questions, see the [README](../README.md) or open a GitHub issue.*
