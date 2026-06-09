# Net-Neutral AI: How It Works

This document explains the architecture of our federated learning prototype, detailing the separation of concerns between the cloud server (Render) and the local clients, data storage, and the execution flow.

## 1. Cloud vs. Local Execution

Our architecture strictly follows a Federated Learning paradigm, ensuring privacy and decentralized computation. 

### What Happens on the Cloud (Render)
Render acts *exclusively* as the **Coordinator / Central Hub**. It does not perform any training. Its responsibilities are:
- **Lobby Management:** Waiting for clients to register and tracking their connection status.
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

- **The Uploaded Dataset:** The Coordinator uploads a single `.csv` file via the frontend dashboard. This file is temporarily saved on Render's ephemeral filesystem. Render performs an 80/20 split:
  - **20% Test Set:** Kept securely on Render (`uploaded_test.csv`) to evaluate the global model.
  - **80% Training Pool:** Sharded and sent to clients. Once sent, Render *does not* use this data.
- **Client Shards:** When a client joins, it downloads its specific shard (e.g., `client_A_data.csv`). This is saved **locally** on the client's hard drive in the `backend/client/local_data/` folder. The client only ever sees its own data, never the full dataset.
- **Leaderboard / Credits Database:** The server maintains an SQLite database (`database.db`) to track how many credits each client earns based on their training contributions. Because Render's free tier uses an ephemeral filesystem, this database resets if the server restarts. (For production, this would sync to a persistent cloud database like Supabase).
- **Model Weights:** The Global Model (`checkpoint.pt`) is saved on Render. Clients download it to a temporary folder, train it, save the new weights locally, upload them to Render, and then delete their local temporary weights.

---

## 3. The Client Agent Setup (CLI Commands)

To participate in the federated training, a client must run the command provided in the **Agent Setup** dashboard. 

**Example Command:**
```bash
python client.py --client_id client_A --coordinator_url https://net-neutral-ai.onrender.com
```

**Why do clients need to run this?**
1. **Repository & Dependencies:** The client must clone the repository and run `pip install -r requirements.txt` so their local machine has the PyTorch libraries required to train a neural network. Render cannot train models on the client's behalf; the client's computer must have the engine.
2. **Establishing Connection:** The `client.py` script is the bridge. It pings the `--coordinator_url` to register itself. 
3. **Continuous Polling:** The script runs an infinite loop, constantly asking Render: *"Is it time for the next round?"*. It automates the entire process of downloading the model, training, and uploading weights, requiring zero human intervention after the initial launch command.
