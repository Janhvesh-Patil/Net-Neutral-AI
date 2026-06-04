// Frontend state
let coordinatorURL = "http://localhost:5000";
let datasetFile = null;
let pollingInterval = null;

/**
 * Set coordinator IP and port
 */
function setCoordinator() {
    const ip = document.getElementById("coordinatorIP").value.trim();
    const port = document.getElementById("coordinatorPort").value.trim();

    if (!ip || !port) {
        showStatus("ipStatus", "Please enter IP and port", "error");
        return;
    }

    coordinatorURL = `http://${ip}:${port}`;
    showStatus("ipStatus", `Coordinator set to: ${coordinatorURL}`, "success");
    logMessage(`[INFO] Coordinator URL: ${coordinatorURL}`, "info");
}

/**
 * Upload dataset to coordinator
 */
async function uploadDataset() {
    const fileInput = document.getElementById("datasetInput");
    const file = fileInput.files[0];

    if (!file) {
        showStatus("uploadStatus", "Please select a CSV file", "error");
        return;
    }

    if (!file.name.toLowerCase().endsWith('.csv')) {
        showStatus("uploadStatus", "Please upload a CSV file", "error");
        return;
    }

    showStatus("uploadStatus", "Uploading...", "pending");
    logMessage(`[INFO] Uploading dataset: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`, "info");

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${coordinatorURL}/upload_dataset`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();
        showStatus("uploadStatus", `Dataset uploaded successfully! (${data.rows} rows)`, "success");
        logMessage(`[SUCCESS] Dataset uploaded with ${data.rows} samples`, "success");
        datasetFile = file.name;

    } catch (error) {
        showStatus("uploadStatus", `Upload failed: ${error.message}`, "error");
        logMessage(`[ERROR] Upload failed: ${error.message}`, "error");
    }
}

/**
 * Refresh list of connected clients
 */
async function refreshClients() {
    try {
        const response = await fetch(`${coordinatorURL}/get_clients`);

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();
        const clients = data.clients || [];

        // Update table
        const tbody = document.querySelector("#clientsTable tbody");
        tbody.innerHTML = "";

        clients.forEach(client => {
            const row = document.createElement("tr");
            const statusClass = client.data_received ? "status-success" : "status-pending";
            const dataReceivedText = client.data_received ? "Yes" : "No";

            row.innerHTML = `
                <td>${client.id}</td>
                <td>${client.ip || "unknown"}</td>
                <td><span class="status-success">Active</span></td>
                <td><span class="${statusClass}">${dataReceivedText}</span></td>
            `;
            tbody.appendChild(row);
        });

        document.getElementById("clientCount").textContent = `Connected: ${clients.length} client(s)`;

    } catch (error) {
        console.error("Failed to refresh clients:", error);
    }
}

/**
 * Start federated training
 */
async function startTraining() {
    if (!datasetFile) {
        showStatus("trainingStatus", "Please upload dataset first", "error");
        return;
    }

    const numClients = parseInt(document.getElementById("numClients").value);

    if (!numClients || numClients < 1) {
        showStatus("trainingStatus", "Enter valid number of clients", "error");
        return;
    }

    showStatus("trainingStatus", "Starting training...", "pending");
    logMessage(`[INFO] Initiating federated training with ${numClients} clients`, "info");

    try {
        const response = await fetch(`${coordinatorURL}/start_training`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ client_count: numClients })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || response.statusText);
        }

        const data = await response.json();
        showStatus("trainingStatus", `Training started! ${data.shards_prepared} data shards prepared`, "success");
        logMessage(`[SUCCESS] ${data.shards_prepared} data shards prepared for clients`, "success");

        // Start polling for updates
        startPolling();

    } catch (error) {
        showStatus("trainingStatus", `Failed to start: ${error.message}`, "error");
        logMessage(`[ERROR] Failed to start training: ${error.message}`, "error");
    }
}

/**
 * Poll for training status updates
 */
function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);

    logMessage(`[INFO] Starting status polling...`, "info");

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${coordinatorURL}/status`);

            if (!response.ok) return;

            const data = await response.json();

            // Log state changes
            if (data.round_status === "done") {
                logMessage(`[SUCCESS] Training complete! Final accuracy: ${(data.global_acc * 100).toFixed(2)}%`, "success");
                clearInterval(pollingInterval);
                showStatus("trainingStatus", "Training Complete", "success");
            } else if (data.round_status === "data_distributing") {
                logMessage(`[INFO] Round ${data.round}: Distributing data to clients...`, "info");
            } else if (data.round_status === "aggregating") {
                logMessage(`[INFO] Round ${data.round}: Aggregating weights...`, "info");
            } else {
                // Log periodically (every 30 seconds to avoid spam)
                if (Math.random() < 0.05) {  // ~5% chance each poll (every ~5 polls)
                    logMessage(`[INFO] Round ${data.round}: Status = ${data.round_status}`, "info");
                }
            }

        } catch (error) {
            logMessage(`[WARNING] Polling error: ${error.message}`, "warning");
        }
    }, 5000);  // Poll every 5 seconds
}

/**
 * Update status message display
 */
function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = message;
        element.className = `status-${type}`;
    }
}

/**
 * Add timestamped message to training log
 */
function logMessage(message, type = "info") {
    const logDiv = document.getElementById("trainingLog");
    if (logDiv) {
        const entry = document.createElement("div");
        entry.className = `log-entry ${type}`;
        const timestamp = new Date().toLocaleTimeString();
        entry.textContent = `[${timestamp}] ${message}`;
        logDiv.appendChild(entry);

        // Auto-scroll to bottom
        logDiv.scrollTop = logDiv.scrollHeight;

        // Limit log entries to last 500 (performance)
        const entries = logDiv.querySelectorAll(".log-entry");
        if (entries.length > 500) {
            entries[0].remove();
        }
    }
}

/**
 * Initialize on page load
 */
document.addEventListener("DOMContentLoaded", () => {
    showStatus("ipStatus", `Coordinator: ${coordinatorURL}`, "success");
    logMessage("[INFO] Frontend loaded successfully", "info");
    logMessage("[INFO] Set coordinator IP and upload dataset to begin", "info");

    // Auto-refresh clients every 3 seconds
    setInterval(refreshClients, 3000);

    // Initial refresh
    refreshClients();
});

/**
 * Handle Enter key in input fields
 */
document.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        if (e.target.id === "coordinatorIP" || e.target.id === "coordinatorPort") {
            setCoordinator();
        } else if (e.target.id === "numClients") {
            startTraining();
        }
    }
});
