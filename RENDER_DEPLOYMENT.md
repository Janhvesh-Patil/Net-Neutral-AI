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
