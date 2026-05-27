"""
config.py — Net-Neutral AI
Shared configuration — edit this file before every demo session.

CRITICAL: Set COORDINATOR_IP to IT's machine IPv4 address.
Run ipconfig on IT's machine and look for IPv4 Address under WiFi adapter.
Example: 192.168.1.42
"""

# ── Network ───────────────────────────────────────────────────────────────────
COORDINATOR_IP = 'IP_ADDRESS_OF_COORDINATOR'   # ← REPLACE with IT machine's local IP
COORDINATOR_PORT = 5000
BASE_URL         = f"http://{COORDINATOR_IP}:{COORDINATOR_PORT}"

# ── Training ──────────────────────────────────────────────────────────────────
TOTAL_ROUNDS  = 5
LOCAL_EPOCHS  = 2
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "../data"          # relative to client/ folder
VOCAB_PATH = "../data/vocab.json"

# ── Client identity ───────────────────────────────────────────────────────────
# Each machine hardcodes its own client ID — do not change after setup
# Machine B (Janhvesh)  : CLIENT_ID = "client_A"
# Machine C (ENTC2)     : CLIENT_ID = "client_B"
# Machine D (ENTC1)     : CLIENT_ID = "client_C"
CLIENT_ID = "client_A"          # ← each machine sets its own value

# ── Coordinator paths ─────────────────────────────────────────────────────────
CHECKPOINT_FILENAME = "checkpoint.pt"
DB_FILENAME         = "database.db"

# ── Timeouts ──────────────────────────────────────────────────────────────────
SUBMISSION_TIMEOUT_SECS = 90    # coordinator waits this long for all clients
POLL_INTERVAL_SECS      = 5     # how often client polls /status between rounds
REGISTER_RETRY_ATTEMPTS = 3     # how many times client retries on connection fail
REGISTER_RETRY_DELAY    = 10    # seconds between retry attempts