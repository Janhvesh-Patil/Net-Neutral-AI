# Net-Neutral AI — Render Deployment Guide

This guide provides step-by-step instructions for deploying the **Net-Neutral AI Coordinator** service on [Render](https://render.com/).

---

## 🚀 Deployment Specifications

Use the following settings when creating a new **Web Service** on Render:

| Field | Setting |
| :--- | :--- |
| **Repository** | `https://github.com/Janhvesh-Patil/Net-Neutral-AI` |
| **Branch** | `scope_creep_update_070626` |
| **Runtime** | `Python` |
| **Build Command** | `pip install -r backend/coordinator/requirements.txt` |
| **Start Command** | `python backend/coordinator/server.py` |

---

## ⚙️ Environment Variables

Configure these variables in the **Environment** tab of your Render Web Service dashboard:

| Variable | Type | Value / Description |
| :--- | :--- | :--- |
| `PORT` | *Automatic* | Render sets this automatically (usually `10000`). The coordinator is pre-configured to dynamically bind to this port. |
| `SUPABASE_URL` | *Optional* | Your Supabase project URL (e.g., `https://xyz.supabase.co`). Required to sync final training round credits to the cloud database. |
| `SUPABASE_KEY` | *Optional* | Your Supabase anon/public API key. Required to sync final training round credits. |

> [!NOTE]
> If `SUPABASE_URL` and `SUPABASE_KEY` are left blank, training will run normally, but leaderboard credits will be stored locally in SQLite (`database.db`) only and will not sync to the cloud database.

---

## 💾 SQLite Database & Ephemeral Storage on Render

Because Render Web Services use an ephemeral filesystem, the local SQLite database (`backend/coordinator/database.db`) and uploaded datasets will be reset whenever the service restarts, redeploys, or goes to sleep (on the free tier).

* **Cloud Persistence**: Enabling Supabase sync (using the environment variables above) ensures that lifetime client credits are safely backed up to the cloud at the completion of a training session.
* **Persistent Disk**: If you require database and checkpoint persistence across restarts without losing local session histories, attach a **Persistent Disk** on Render (available on paid tiers) and mount it.

---

## 🖥️ Using the Frontend Dashboard

Once the app is deployed, open the Render public URL (e.g., `https://your-app-name.onrender.com`).

* **Auto-Connection**: The Web UI automatically detects that it is hosted and will default its target address to `window.location.origin`. You can leave the **IP Address** and **Port** inputs blank on the setup screen and click **Launch Coordinator Dashboard** or **Connect** to proceed!

---

## 🔌 Connecting Local Python Clients

To connect local training client nodes to the cloud coordinator on Render, pass the Render URL using the `COORDINATOR_BASE_URL` environment variable before starting the client:

### Command Prompt (Windows CMD)
```cmd
set COORDINATOR_BASE_URL=https://your-app-name.onrender.com
python client.py --client_id client_A
```

### PowerShell (Windows)
```powershell
$env:COORDINATOR_BASE_URL="https://your-app-name.onrender.com"
python client.py --client_id client_A
```

### Bash (Linux / macOS / Git Bash)
```bash
export COORDINATOR_BASE_URL="https://your-app-name.onrender.com"
python client.py --client_id client_A
```
