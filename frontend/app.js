/* ============================================================
   Net-Neutral AI — Frontend Logic
   Handles screen navigation, coordinator mode, and client mode.
   ============================================================ */

// ── Global State ────────────────────────────────────────────────
let coordinatorURL = "https://net-neutral-ai.onrender.com:5000";
let currentRole = null;          // 'coordinator' or 'client'
let clientName = null;           // e.g. 'client_A'
let expectedClients = 3;
let datasetFile = null;
let pollingInterval = null;


// ── Screen Navigation ───────────────────────────────────────────

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
}

function selectRole(role) {
    currentRole = role;
    if (role === 'coordinator') {
        showScreen('screen-coordinator-setup');
    } else {
        showScreen('screen-client-setup');
    }
}

function goBack() {
    // Stop any polling
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
    showScreen('screen-role');
}


// ── Coordinator Setup ───────────────────────────────────────────

function launchCoordinator() {
    const ip = document.getElementById('coord-ip').value.trim();
    const port = document.getElementById('coord-port').value.trim();
    expectedClients = parseInt(document.getElementById('coord-clients').value);

    if (!ip) {
        coordinatorURL = window.location.origin;
    } else if (ip.includes('://')) {
        coordinatorURL = ip;
    } else {
        coordinatorURL = `http://${ip}${port ? ':' + port : ''}`;
    }

    document.getElementById('coord-url-badge').textContent = coordinatorURL.replace(/^https?:\/\//, '');

    showScreen('screen-coordinator-dashboard');
    logMessage('trainingLog', `[INFO] Coordinator dashboard loaded`, 'info');
    logMessage('trainingLog', `[INFO] URL: ${coordinatorURL}`, 'info');
    logMessage('trainingLog', `[INFO] Expected clients: ${expectedClients}`, 'info');

    // Start auto-refresh of clients
    refreshClients();
    setInterval(refreshClients, 3000);
}


// ── Client Setup ────────────────────────────────────────────────

async function connectAsClient() {
    const ip = document.getElementById('client-coord-ip').value.trim();
    const port = document.getElementById('client-coord-port').value.trim();
    clientName = document.getElementById('client-name').value;

    if (!ip) {
        coordinatorURL = window.location.origin;
    } else if (ip.includes('://')) {
        coordinatorURL = ip;
    } else {
        coordinatorURL = `http://${ip}${port ? ':' + port : ''}`;
    }

    showStatus('client-setup-status', 'Connecting...', 'pending');

    try {
        // Register with coordinator
        const response = await fetch(`${coordinatorURL}/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                client_id: clientName,
                ip_address: 'web-frontend'
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || response.statusText);
        }

        // Success — show client dashboard
        document.getElementById('client-dashboard-title').textContent = `${clientName.replace('_', ' ').toUpperCase()}`;
        document.getElementById('client-name-badge').textContent = clientName;
        document.getElementById('client-coord-url').textContent = coordinatorURL;
        document.getElementById('client-conn-status').textContent = 'Connected';
        document.getElementById('client-conn-status').className = 'stat-value status-success';

        showScreen('screen-client-dashboard');
        logMessage('clientLog', `[OK] Registered with coordinator as ${clientName}`, 'success');
        logMessage('clientLog', `[INFO] Coordinator: ${coordinatorURL}`, 'info');

        // Start polling for updates
        startClientPolling();

    } catch (error) {
        showStatus('client-setup-status', `Connection failed: ${error.message}`, 'error');
    }
}


// ── Client Dashboard Polling ────────────────────────────────────

function startClientPolling() {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${coordinatorURL}/api/client_status/${clientName}`);

            if (!response.ok) return;
            const data = await response.json();

            document.getElementById('client-round').textContent =
                `${data.current_round} / ${data.total_rounds}`;
            document.getElementById('client-round-status').textContent =
                formatStatus(data.round_status);
            document.getElementById('client-credits').textContent =
                data.total_credits.toLocaleString();
            document.getElementById('client-accuracy').textContent =
                data.global_accuracy > 0 ? (data.global_accuracy * 100).toFixed(2) + '%' : '--';

            if (data.round_status === 'done') {
                logMessage('clientLog', `[OK] Training complete! Final credits: ${data.total_credits}`, 'success');
                clearInterval(pollingInterval);
            }

        } catch (error) {
            // Silent fail on poll errors
        }
    }, 3000);
}

function formatStatus(status) {
    const map = {
        'waiting_for_clients': 'Waiting for clients',
        'data_distributing': 'Distributing data',
        'active': 'Training active',
        'aggregating': 'Aggregating weights',
        'done': 'Complete'
    };
    return map[status] || status;
}


// ── Coordinator: Upload Dataset ─────────────────────────────────

async function uploadDataset() {
    const fileInput = document.getElementById('datasetInput');
    const file = fileInput.files[0];

    if (!file) {
        showStatus('uploadStatus', 'Please select a CSV file', 'error');
        return;
    }

    if (!file.name.toLowerCase().endsWith('.csv')) {
        showStatus('uploadStatus', 'File must be .csv format', 'error');
        return;
    }

    showStatus('uploadStatus', 'Uploading...', 'pending');
    logMessage('trainingLog', `[INFO] Uploading: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`, 'info');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${coordinatorURL}/upload_dataset`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`Server error: ${response.statusText}`);

        const data = await response.json();
        showStatus('uploadStatus', `Uploaded! ${data.rows} samples`, 'success');
        logMessage('trainingLog', `[OK] Dataset uploaded: ${data.rows} samples`, 'success');
        datasetFile = file.name;

    } catch (error) {
        showStatus('uploadStatus', `Failed: ${error.message}`, 'error');
        logMessage('trainingLog', `[ERROR] Upload failed: ${error.message}`, 'error');
    }
}


// ── Coordinator: Refresh Clients ────────────────────────────────

async function refreshClients() {
    try {
        const response = await fetch(`${coordinatorURL}/get_clients`);
        if (!response.ok) return;

        const data = await response.json();
        const clients = data.clients || [];

        const tbody = document.querySelector('#clientsTable tbody');
        tbody.innerHTML = '';

        clients.forEach(client => {
            const row = document.createElement('tr');
            const statusClass = client.data_received ? 'status-success' : 'status-pending';
            row.innerHTML = `
                <td>${client.id}</td>
                <td>${client.ip || 'unknown'}</td>
                <td><span class="status-success">Active</span></td>
                <td><span class="${statusClass}">${client.data_received ? 'Yes' : 'No'}</span></td>
            `;
            tbody.appendChild(row);
        });

        document.getElementById('clientCount').textContent = `Connected: ${clients.length} client(s)`;

    } catch (error) {
        // Silent fail during polling
    }
}


// ── Coordinator: Start Training ─────────────────────────────────

async function startTraining() {
    if (!datasetFile) {
        showStatus('trainingStatus', 'Upload a dataset first', 'error');
        return;
    }

    showStatus('trainingStatus', 'Starting...', 'pending');
    logMessage('trainingLog', `[INFO] Starting training with ${expectedClients} client(s)`, 'info');

    try {
        const response = await fetch(`${coordinatorURL}/start_training`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ client_count: expectedClients })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || response.statusText);
        }

        const data = await response.json();
        showStatus('trainingStatus', `Training started! ${data.shards_prepared} shards prepared`, 'success');
        logMessage('trainingLog', `[OK] ${data.shards_prepared} data shards prepared`, 'success');

        // Start status polling
        startCoordinatorPolling();

    } catch (error) {
        showStatus('trainingStatus', `Failed: ${error.message}`, 'error');
        logMessage('trainingLog', `[ERROR] ${error.message}`, 'error');
    }
}


// ── Coordinator: Status Polling ─────────────────────────────────

function startCoordinatorPolling() {
    if (pollingInterval) clearInterval(pollingInterval);

    let lastRound = 0;
    let lastStatus = '';

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${coordinatorURL}/status`);
            if (!response.ok) return;

            const data = await response.json();

            // Log state changes (not every poll)
            if (data.round !== lastRound || data.round_status !== lastStatus) {
                if (data.round_status === 'done') {
                    logMessage('trainingLog', `[OK] Training complete!`, 'success');
                    showStatus('trainingStatus', 'Training Complete!', 'success');
                    clearInterval(pollingInterval);
                } else if (data.round_status === 'aggregating') {
                    logMessage('trainingLog', `[INFO] Round ${data.round}: Aggregating...`, 'info');
                } else if (data.round_status === 'active') {
                    logMessage('trainingLog', `[INFO] Round ${data.round}: Training active`, 'info');
                } else if (data.round_status === 'data_distributing') {
                    logMessage('trainingLog', `[INFO] Distributing data to clients...`, 'info');
                }

                lastRound = data.round;
                lastStatus = data.round_status;
            }

        } catch (error) {
            logMessage('trainingLog', `[WARN] Poll error: ${error.message}`, 'warning');
        }
    }, 5000);
}


// ── Helpers ─────────────────────────────────────────────────────

function showStatus(elementId, message, type) {
    const el = document.getElementById(elementId);
    if (el) {
        el.textContent = message;
        el.className = `status-${type}`;
    }
}

function logMessage(logId, message, type) {
    const logDiv = document.getElementById(logId);
    if (!logDiv) return;

    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const ts = new Date().toLocaleTimeString();
    entry.textContent = `[${ts}] ${message}`;
    logDiv.appendChild(entry);
    logDiv.scrollTop = logDiv.scrollHeight;

    // Cap at 500 entries
    const entries = logDiv.querySelectorAll('.log-entry');
    if (entries.length > 500) entries[0].remove();
}


// ── Init ────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Start on role selection screen
    showScreen('screen-role');
});
