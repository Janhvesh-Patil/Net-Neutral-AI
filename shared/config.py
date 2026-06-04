"""
CRITICAL: Set COORDINATOR_IP
Run ipconfig on coordinator machine and look for IPv4 Address under WiFi adapter.
Example: 192.168.1.42
"""

# ── Network ───────────────────────────────────────────────────────────────────
COORDINATOR_IP = 'IP_ADDRESS_OF_COORDINATOR'
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
CLIENT_ID = "client_A"

# ── Coordinator paths ─────────────────────────────────────────────────────────
CHECKPOINT_FILENAME = "checkpoint.pt"
DB_FILENAME         = "database.db"

# ── Timeouts ──────────────────────────────────────────────────────────────────
SUBMISSION_TIMEOUT_SECS = 90    # coordinator waits this long for all clients
POLL_INTERVAL_SECS      = 5     # how often client polls /status between rounds
REGISTER_RETRY_ATTEMPTS = 3     # how many times client retries on connection fail
REGISTER_RETRY_DELAY    = 10    # seconds between retry attempts

# ── Data Distribution (NEW) ───────────────────────────────────────────────────
WAIT_FOR_DATA_TIMEOUT_SECS = 300  # how long client waits for data before timeout
LOCAL_DATA_DIR = "local_data"     # where clients store their data shard
DATA_SHARD_FILENAME = "{client_id}_data.csv"  # format for shard filename