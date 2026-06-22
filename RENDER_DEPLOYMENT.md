# Net-Neutral AI — Render Deployment Guide

This guide provides step-by-step instructions for deploying the **Net-Neutral AI Coordinator** service on [Render](https://render.com/).

---

## 🚀 Deployment Specifications

Use the following settings when creating a new **Web Service** on Render:

| Field | Setting |
| :--- | :--- |
| **Repository** | `https://github.com/Janhvesh-Patil/Net-Neutral-AI` |
| **Branch** | `testing_site` |
| **Runtime** | `Python` |
| **Build Command** | `pip install -r backend/coordinator/requirements-render.txt` |
| **Start Command** | `python backend/coordinator/server.py` |

---

## ⚙️ Environment Variables

Configure these variables in the **Environment** tab of your Render Web Service dashboard:

| Variable | Type | Value / Description |
| :--- | :--- | :--- |
| `PORT` | *Automatic* | Render sets this automatically (usually `10000`). The coordinator is pre-configured to dynamically bind to this port. |

---

## 💾 SQLite Database & Ephemeral Storage on Render

Because Render Web Services use an ephemeral filesystem, the local SQLite database (`backend/coordinator/database.db`) and uploaded datasets will be reset whenever the service restarts, redeploys, or goes to sleep (on the free tier).

* **Persistent Disk**: If you require database and checkpoint persistence across restarts without losing local session histories, attach a **Persistent Disk** on Render (available on paid tiers) and mount it.

---

## 🖥️ Using the Frontend Dashboard

Once the app is deployed, open the Render public URL (e.g., `https://your-app-name.onrender.com`).

* **Auto-Connection**: The Web UI automatically detects that it is hosted and will dynamically route all traffic through `window.location.origin`. There are no IP Address or Port inputs to configure! Just click **Launch Coordinator Dashboard** or **Connect** to proceed!

---

## 🔌 Connecting Local Python Clients

To connect local training client nodes to the cloud coordinator on Render, simply pass the Render URL using the `--coordinator_url` argument when starting the client. The frontend provides copy-paste commands, but you can also run them manually:

### Command Prompt (Windows CMD)
```cmd
python client.py --client_id client_A --coordinator_url https://your-app-name.onrender.com
```

### PowerShell (Windows)
```powershell
python client.py --client_id client_A --coordinator_url https://your-app-name.onrender.com
```

### Bash (Linux / macOS / Git Bash)
```bash
python3 client.py --client_id client_A --coordinator_url https://your-app-name.onrender.com
```

---

## 🛠️ Step-by-Step: Deployment & Single-Client Testing

### Phase 1: Deploying the Coordinator on Render
1. **Create a New Web Service**: Log in to [Render.com](https://render.com) and click **New > Web Service**.
2. **Connect Repository**: Select your GitHub repository (`Janhvesh-Patil/Net-Neutral-AI`).
3. **Configure Settings**:
   - **Runtime**: Python
   - **Build Command**: `pip install -r backend/coordinator/requirements-render.txt`
   - **Start Command**: `python backend/coordinator/server.py`
4. **Deploy**: Click **Create Web Service**. Render will install the dependencies, automatically set a `PORT` environment variable, and start the server. 
5. **Access the Website**: Once the deployment turns green, click the URL provided by Render (e.g., `https://net-neutral-ai.onrender.com`). The frontend dashboard will load.

### Phase 2: Testing with One Client System
To do a real-world test connecting a local machine to your newly deployed cloud coordinator:

1. **Launch the Coordinator Lobby**:
   - On your Render website, go to the **Coordinator Setup** screen. 
   - Change the "Expected Clients" input to `1` (since you're doing a single-client test).
   - Click **Launch Coordinator Dashboard**. The server is now actively waiting for 1 client to join.
2. **Upload Dataset**: On the resulting Dashboard screen, click the **Upload Dataset** button. Upload a `.csv` file. **Note:** The CSV must strictly contain a `review` column (text) and a `label` column (0 or 1) for the model to process it.
3. **Start the Local Client**:
   - On the machine you want to use as a client, open a terminal/command prompt.
   - Clone your repo and navigate to `backend/client`.
   - Install dependencies if you haven't: `pip install -r requirements.txt`.
   - Run the client command pointing to your Render URL:
     ```bash
     python client.py --client_id my_test_client --coordinator_url https://your-app-name.onrender.com
     \`\`\`
4. **Begin Training**:
   - Your local terminal will output that it successfully connected to the coordinator.
   - Look back at the Render website. You will see `my_test_client` appear in the connected devices list.
   - Click the **Start Training** button on the website.
   - Your local terminal will automatically download the data shard and model weights, train the model, and upload the updated weights back to the Render server!
