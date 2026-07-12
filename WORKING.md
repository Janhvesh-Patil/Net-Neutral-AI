# Net-Neutral AI: How It Works

This document explains the architecture of our federated learning prototype, detailing the separation of concerns between the central server and the local clients, data storage, and the execution flow.

## 1. Cloud vs. Local Execution

Our architecture strictly follows a Federated Learning paradigm, ensuring privacy and decentralized computation. 

### What Happens on the Coordinator Server
The Coordinator can be deployed to the cloud (e.g., Render) OR run entirely on a local, high-performance machine via **Local Coordinator Mode** to bypass cloud memory limitations. The Coordinator does not perform any training. Its responsibilities are:
- **Lobby Management:** Waiting for clients to register and tracking their connection status via real-time Server-Sent Events (SSE).
- **Data Distribution:** Receiving the initial dataset from the Coordinator dashboard, splitting it 80/20, sharding the 80% into smaller chunks, and distributing them to registered clients.
- **Federated Averaging (FedAvg):** Receiving trained model weights from all clients at the end of each round, mathematically averaging them together to create a new "Global Model," and evaluating its accuracy.
- **State Machine:** Managing the global round clock (e.g., waiting for all clients to submit round 1 weights before broadcasting round 2).

### What Happens Locally (Client Devices)
The client machines do the heavy lifting. Their responsibilities are:
- **Data Storage:** Downloading their specific "shard" of the training data from the Coordinator and saving it locally.
- **Model Training:** Downloading the latest Global Model, loading their local data shard into memory, and running PyTorch backpropagation (training) using their local CPU/GPU.
- **Weight Submission:** Exporting the newly trained weights and securely uploading them back to the Coordinator.

---

## 2. How Data is Saved & Stored

Data privacy is the core tenet of this system.

- **The Uploaded Dataset:** The Coordinator uploads a single `.csv` file via the frontend dashboard. This file is temporarily saved. The Coordinator performs an 80/20 split:
  - **20% Test Set:** Kept securely on the Coordinator (`uploaded_test.csv`) to evaluate the global model.
  - **80% Training Pool:** Sharded and sent to clients. Once sent, the Coordinator *does not* use this data.
- **Client Shards:** When a client joins, it downloads its specific shard (e.g., `client_A_data.csv`). This is saved **locally** on the client's hard drive in the `backend/client/local_data/` folder. The client only ever sees its own data, never the full dataset.
- **Leaderboard / Credits Database:** The server maintains an SQLite database (`database.db`) to track how many credits each client earns based on their training contributions.
- **Model Weights & Deployment:** The Global Model (`checkpoint.pt`) is saved on the Coordinator. Clients download it to a temporary folder, train it, save the new weights locally, upload them, and delete their local temporary weights. Once training is complete, the final global model can be seamlessly exported via the **Download Global Model (.pt)** feature on the dashboard for external deployment.

---

## 3. The Agent Setup (CLI Commands)

To participate in the federated network, users must run the background agent commands provided in the **Agent Setup** dashboard. The setup guide dynamicly adapts to the user's role (Client vs. Coordinator).

**Example Client Command:**
```bash
python client.py --client_id client_A --coordinator_url https://net-neutral-ai.onrender.com
```

**Example Coordinator Command (Local Coordinator Mode):**
```bash
python server.py
```

**Why do users need to run this?**
1. **Repository & Dependencies:** The machine must clone the repository and run `pip install -r requirements.txt` so it has the Python and PyTorch libraries required to train or aggregate the neural network. The web browser cannot train models natively; the computer must run the backend engine.
2. **Establishing Connection:** For clients, the `client.py` script acts as the bridge. It pings the `--coordinator_url` to register itself. 
3. **Continuous Polling & Real-Time Sync:** The client script runs an infinite loop, constantly asking the server: *"Is it time for the next round?"*. Combined with the frontend's Server-Sent Events (SSE) tracking Live Epoch Metrics, the entire process is automated and visible in real-time, requiring zero human intervention after the initial launch command.

---

## 4. Real-Time UI & Dynamic Configurations

The frontend application has been upgraded to support full real-time synchronisation and dynamic tracking:
- **Dynamic Training Configurations:** The Coordinator can now specify the total number of rounds and epochs directly from the setup dashboard, which dynamically drives the server's federated learning loops.
- **HD Network Topology:** A high-definition, dynamic canvas tracks live connection states between the Coordinator and active Clients.
- **Coordinator Dashboard & Tabbed Epoch Metrics:** Instead of a scattered view, the Coordinator organizes real-time epoch metrics into tabbed views per round, providing an elegant and scalable tabular UI to monitor training loss and accuracy.
- **Client Timeline & Credits Sync:** The Client Dashboard automatically rebuilds its historical timeline from the server API, guaranteeing that credits, points earned, and round statuses accurately persist across page refreshes and late joins.
