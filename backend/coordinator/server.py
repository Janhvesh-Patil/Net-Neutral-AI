import os
import glob
import json
import queue
import datetime
import torch
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

# Import internal modules
from . import fedavg
from . import credits
from . import evaluate
from . import data_distributor
import threading

# Global state lock for thread safety
state_lock = threading.RLock()

# --- SSE (Server-Sent Events) Infrastructure ---
# Each connected browser gets its own queue; broadcast pushes to all.
sse_clients = []   # list of queue.Queue
sse_lock = threading.Lock()

def broadcast_event(event_type: str, data: dict) -> None:
    """Push an SSE event to every connected browser."""
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    dead = []
    with sse_lock:
        for q in sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)

# In-memory store for per-epoch training data reported by clients
epoch_reports = {}   # {client_id: [{epoch, loss, accuracy, samples, round}]}

# Micro-states for detailed UI animations (downloading, training, uploading, idle)
client_micro_states = {} # {client_id: state_string}

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'frontend'),
    static_url_path=''
)
CORS(app)

# --- DATABASE INITIALIZATION ---
# Initialize database at app startup (critical for both __main__ and production WSGI)
credits.init_db()

# --- PATH CONFIGURATION ---
# Ensures the server always looks in the exact folder where server.py lives
COORDINATOR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(COORDINATOR_DIR, 'checkpoint.pt')

# --- Global State Machine ---
current_round = 1
TOTAL_ROUNDS = 5
LOCAL_EPOCHS = 2
registered_clients = set()
round_status = 'waiting_for_clients'
global_accuracy = 0.0
accuracy_history = []  # List of dicts: [{'round': r, 'accuracy': acc}]
round_start_time = datetime.datetime.now()

# Dictionaries to track submissions
submitted_weights = {}
submitted_samples = {}

# Data distribution
# NOTE: client_registry is kept in-memory ONLY for the session-scoped 'data_received'
# flag. Client identity (id, ip, registered_at) is now persisted in the SQLite
# clients table via credits.register_client().
client_registry = {}  # {client_id: {'data_received': False}} — session-scoped only
data_shards = {}      # {client_id: DataFrame} - in-memory cache of divided data
uploaded_dataset_path = os.path.join(COORDINATOR_DIR, 'uploaded_dataset.csv')
model_downloaded = False  # Flag to track if coordinator has downloaded the final model

# Cache for evaluation
_cached_val_loader = None

# --- Evaluation Wrapper ---
def run_evaluation_from_path(model_path: str, round_number: int) -> float:
    global _cached_val_loader
    import sys
    # Ensure Python can find the client folder
    project_root = os.path.dirname(COORDINATOR_DIR)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from client.model import TransformerClassifier
    from client.data import Vocabulary, get_validation_dataloader
    import data_distributor
    
    if _cached_val_loader is None:
        print("[Coordinator] Initializing validation dataloader from cached data...")
        test_csv_path = os.path.join(COORDINATOR_DIR, "uploaded_test.csv")
        global_vocab_path = os.path.join(COORDINATOR_DIR, "vocab.json")
        
        if not os.path.exists(global_vocab_path):
            raise FileNotFoundError("global vocab.json not found. Was dataset uploaded?")
            
        vocab = Vocabulary.load(global_vocab_path)
        df_test = data_distributor.load_and_validate_csv(test_csv_path)
        test_texts = df_test['review'].tolist()
        test_labels = df_test['label'].tolist()
        _cached_val_loader = get_validation_dataloader(test_texts, test_labels, vocab)
    
    val_loader = _cached_val_loader
    
    # 2. Initialize an empty model and load the weights from the file
    model = TransformerClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    print("\n[Coordinator] Running global evaluation...")
    result = evaluate.evaluate(model, val_loader)
    
    # 4. Print the formatted results to the terminal
    prev_accuracy = credits._get_previous_accuracy(round_number, credits.DB_PATH) if round_number > 1 else 0.0
    print(evaluate.format_eval_result(result, round_num=round_number, total_rounds=TOTAL_ROUNDS, prev_accuracy=prev_accuracy))
    
    # Free memory to prevent OOM on 512MB limit
    accuracy = result.accuracy
    del model
    del result
    import gc
    gc.collect()
    
    return accuracy

# --- Core Aggregation Logic ---
def _process_round_completion(current_round_copy, round_start_time_copy, weights_copy, samples_copy):
    global current_round, round_status, global_accuracy, round_start_time

    print(f"\n--- All clients submitted for round {current_round_copy}. Running FedAvg ---")
    broadcast_event('sys_log', {'message': f'Received weights from all clients. Transferring to aggregator for round {current_round_copy}...', 'level': 'info'})

    client_weights = {}
    for cid, fpath in weights_copy.items():
        state_dict, err = fedavg.load_client_weights(fpath, cid)
        if not err:
            client_weights[cid] = state_dict
        else:
            print(f"[Coordinator] ⚠ Error loading {cid}: {err}")

    try:
        # 1. Run FedAvg
        broadcast_event('sys_log', {'message': 'Starting Federated Averaging (FedAvg)...', 'level': 'info'})
        result = fedavg.federated_average(client_weights, client_samples=samples_copy)

        # 2. Save using the absolute MODEL_PATH
        fedavg.save_global_model(result.global_state_dict, MODEL_PATH)
        broadcast_event('sys_log', {'message': 'FedAvg complete. New global model generated.', 'level': 'info'})

        # 3. Evaluate the new global model
        global_acc = run_evaluation_from_path(MODEL_PATH, current_round_copy)

        # 4. Log the round and print leaderboard
        credits.log_round(current_round_copy, round_start_time_copy, result.clients_included, global_acc)
        board = credits.get_leaderboard()

        # Broadcast accuracy update via SSE
        broadcast_event('accuracy_update', {
            'round': current_round_copy,
            'accuracy': global_acc,
            'clients_submitted': result.clients_included,
        })

        if current_round_copy >= TOTAL_ROUNDS:
            print(credits.format_final_leaderboard(board))
        else:
            print(credits.format_leaderboard(board, current_round_copy))

        # Update global accuracy safely
        with state_lock:
            global_accuracy = global_acc
            accuracy_history.append({
                'round': current_round_copy,
                'accuracy': global_acc
            })

    except fedavg.FedAvgError as e:
        print(f"[Coordinator] ⚠ FedAvg failed: {e}")
    except Exception as e:
        print(f"[Coordinator] ⚠ Evaluation failed: {e}")

    # 5. Clean up temporary weight files
    for fpath in list(weights_copy.values()):
        if os.path.exists(fpath):
            os.remove(fpath)

    # 6. Step the state machine forward
    if current_round_copy >= TOTAL_ROUNDS:
        with state_lock:
            round_status = 'done'
        print("\n[Coordinator] Training complete! Final leaderboard ready.")
        broadcast_event('training_done', {
            'final_accuracy': global_acc,
            'total_rounds': TOTAL_ROUNDS,
        })
        print("[Coordinator] Training session complete! Resetting active round state.")
        data_distributor.is_active = False
    else:
        with state_lock:
            current_round += 1
            round_status = 'active'
            round_start_time = datetime.datetime.now()
        print(f"\n[Coordinator] Ready for Round {current_round}")
        broadcast_event('round_start', {
            'round': current_round,
            'status': 'active',
        })

def check_round_completion():
    global current_round, round_status, global_accuracy, round_start_time

    # Only block aggregation if we haven't started distributing data yet
    if round_status in ['waiting_for_clients']:
        return

    if len(submitted_weights) >= len(registered_clients) > 0:
        round_status = 'aggregating'
        
        # Snapshot state for the background thread
        current_round_copy = current_round
        round_start_time_copy = round_start_time
        weights_copy = submitted_weights.copy()
        samples_copy = submitted_samples.copy()
        
        # Clear globals immediately so they are ready for the next round
        submitted_weights.clear()
        submitted_samples.clear()
        
        # Start background thread
        threading.Thread(
            target=_process_round_completion,
            args=(current_round_copy, round_start_time_copy, weights_copy, samples_copy),
            daemon=True
        ).start()

# --- API Endpoints ---

@app.route('/api/config', methods=['GET'])
def api_config():
    """Returns the coordinator's current training configuration.
    Called by clients at startup AND after shard download to get
    the correct TOTAL_ROUNDS and LOCAL_EPOCHS set by /start_training.
    """
    return jsonify({
        'total_rounds':  TOTAL_ROUNDS,
        'local_epochs':  LOCAL_EPOCHS,
        'epochs':        LOCAL_EPOCHS,  # alias for older clients
        'batch_size':    32,
        'learning_rate': 1e-3,
    })


@app.route('/register', methods=['POST'])
def register():
    """Called once per client at startup. Persists to SQLite clients table."""
    data = request.get_json()
    client_id = data.get('client_id')
    client_ip = data.get('ip_address', 'unknown')

    if not client_id:
        return jsonify({'error': 'client_id missing'}), 400

    # Persist client identity to database
    credits.register_client(client_id, client_ip)
    registered_clients.add(client_id)

    # Session-scoped tracking for data distribution
    if client_id not in client_registry:
        client_registry[client_id] = {'data_received': False}

    print(f"[Coordinator] Node Registered: {client_id} from {client_ip}")

    # Broadcast client join via SSE
    broadcast_event('client_joined', {
        'client_id': client_id,
        'ip': client_ip,
        'total_clients': len(registered_clients),
    })

    return jsonify({'status': 'ok', 'round': current_round})

@app.route('/model', methods=['GET'])
def get_model():
    """Serves the binary .pt global model file [cite: 1, 107-108]."""
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not available'}), 503
    return send_file(MODEL_PATH, as_attachment=True)

@app.route('/submit', methods=['POST'])
def submit():
    """Receives local weights and metadata via multipart form-data [cite: 1, 113-115]."""
    global round_status, submitted_weights, submitted_samples
    
    client_id = request.form['client_id']
    if  not client_id:
        return jsonify({'error': 'client_id missing'}), 400
    
    weights_file = request.files['weights']
    if not weights_file:
        return jsonify({'error': 'weights file missing'}), 400

    with state_lock:
        try:
            samples_trained = int(request.form.get('samples_trained', 0))
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid samples_trained'}), 400

        # BUG-06: Use .get() to avoid KeyError if field is missing,
        # which would lock state_lock permanently.
        try:
            time_seconds = float(request.form.get('time_seconds', 0))
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid time_seconds'}), 400

        # Safety fix: if weights arrive while status is still data_distributing
        # (e.g. shard was fetched but transition didn't fire), unblock the state machine.
        if round_status == 'data_distributing':
            round_status = 'active'
            print(f"[Coordinator] Safety: submit received while data_distributing. Status → active.")
    
    
        # Save the binary weight file temporarily in the absolute dir [cite: 1, 117-118]
        uploaded_filename = weights_file.filename or 'weights.pt'
        save_path = os.path.join(COORDINATOR_DIR, f"temp_{client_id}_round{current_round}.pt")
        raw_path  = save_path + ('.gz' if uploaded_filename.endswith('.gz') else '')
        try:
            weights_file.save(raw_path)

            # Decompress gzip if the client sent a compressed file
            if raw_path.endswith('.gz'):
                import gzip, shutil
                with gzip.open(raw_path, 'rb') as f_in, open(save_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(raw_path)

            if not os.path.exists(save_path):
                return jsonify({'error': 'Failed to save weights'}), 500
            
            submitted_weights[client_id] = save_path
            submitted_samples[client_id] = samples_trained
        except Exception as e:
            for p in [save_path, raw_path]:
                if os.path.exists(p):
                    os.remove(p)
            return jsonify({'error': f'File upload failed: {str(e)}'}), 400

        
        # FIX 1.3: Ensure the round row exists before inserting credits (FK ordering)
        credits.ensure_round_exists(current_round, round_start_time)
        
        # Log to SQLite
        points_earned = credits.log_credit(client_id, current_round, samples_trained, time_seconds)
        
        check_round_completion()
    
    return jsonify({
        'credits': points_earned,
        'round': current_round,
        'global_acc': global_accuracy
    })

@app.route('/status', methods=['GET'])
def status():
    """Polled every 3 seconds by clients to check state machine."""
    # Include leaderboard data for the live training view
    lb = credits.get_leaderboard_dicts()
    submitted_this_round = list(submitted_weights.keys())
    return jsonify({
        'round': current_round,
        'round_status': round_status,
        'active_clients': list(registered_clients),
        'clients_submitted': submitted_this_round,
        'global_accuracy': global_accuracy,
        'accuracy_history': accuracy_history,
        'total_rounds': TOTAL_ROUNDS,
        'leaderboard': lb,
        'client_micro_states': client_micro_states,
        'clients': [
            {'id': cid, 'ip': credits.get_client(cid).ip_address if credits.get_client(cid) else 'unknown'}
            for cid in registered_clients
        ],
        'epoch_reports': epoch_reports,
    })

@app.route('/results', methods=['GET'])
def results():
    """Returns final dashboard stats."""
    return jsonify({
        'final_accuracy': global_accuracy,
        'total_rounds_completed': current_round if round_status == 'done' else current_round - 1
    })


@app.route('/api/client_status/<client_id>', methods=['GET'])
def api_client_status(client_id):
    """Returns individual client's credits, round participation, and history."""
    client = credits.get_client(client_id)
    if client is None:
        return jsonify({'error': f'Client {client_id} not found'}), 404

    total_credits = credits.get_client_total_credits(client_id)
    submitted = credits.get_submitted_clients(current_round)
    round_history = credits.get_round_history(client_id)

    # Map coordinator-level statuses that don't make sense for individual clients
    _client_status_map = {
        'data_distributing':   'idle',
        'waiting_for_clients': 'idle',
    }
    micro = client_micro_states.get(client_id)
    if micro is None:
        micro = _client_status_map.get(round_status, round_status)

    return jsonify({
        'client_id': client_id,
        'ip_address': client.ip_address,
        'registered_at': client.registered_at,
        'total_credits': total_credits,
        'current_round': current_round,
        'round_status': round_status,
        'micro_status': micro,
        'total_rounds': TOTAL_ROUNDS,
        'has_submitted_this_round': client_id in submitted,
        'global_accuracy': global_accuracy,
        'round_history': round_history,
        'epoch_reports': epoch_reports.get(client_id, []),
    })

@app.route('/api/client_state', methods=['POST'])
def api_client_state():
    """Updates a client's specific micro-state (e.g. downloading, training, uploading)."""
    data = request.get_json() or {}
    client_id = data.get('client_id')
    state = data.get('state')
    
    if not client_id or not state:
        return jsonify({'error': 'Missing client_id or state'}), 400
        
    client_micro_states[client_id] = state
    return jsonify({'status': 'ok'})


# --- NEW ENDPOINTS FOR DATA DISTRIBUTION ---

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """Accept CSV dataset upload from frontend."""
    global round_status

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
        return jsonify({'error': 'Invalid filename'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be CSV format'}), 400

    try:
        # Save uploaded file
        file.save(uploaded_dataset_path)
        print(f"[Coordinator] Dataset uploaded: {uploaded_dataset_path}")

        # Validate the dataset
        df = data_distributor.load_and_validate_csv(uploaded_dataset_path)
        num_rows = len(df)

        # Build and save global vocabulary
        project_root = os.path.dirname(COORDINATOR_DIR)
        import sys
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from client.data import Vocabulary, VOCAB_SIZE, TRAIN_SIZE
        
        global_vocab_path = os.path.join(COORDINATOR_DIR, "vocab.json")
        vocab = Vocabulary()
        vocab.build(df['review'].tolist()[:TRAIN_SIZE], max_size=VOCAB_SIZE)
        vocab.save(global_vocab_path)
        print(f"[Coordinator] Global vocabulary built: {vocab.size} tokens")

        print(f"[Coordinator] Dataset validated: {num_rows} samples")
        return jsonify({'status': 'ok', 'rows': num_rows})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/get_clients', methods=['GET'])
def get_clients():
    """Return list of registered clients — persistent data from DB, session flags from memory."""
    db_clients = credits.get_all_clients()
    clients_list = [
        {
            'id': c.client_id,
            'ip': c.ip_address,
            'registered_at': c.registered_at,
            'data_received': client_registry.get(c.client_id, {}).get('data_received', False)
        }
        for c in db_clients
    ]
    return jsonify({'clients': clients_list, 'count': len(clients_list)})


@app.route('/vocab', methods=['GET'])
def get_vocab():
    """Serve the globally generated vocabulary JSON file."""
    vocab_path = os.path.join(COORDINATOR_DIR, 'vocab.json')
    if os.path.exists(vocab_path):
        return send_file(vocab_path, mimetype='application/json')
    return jsonify({'error': 'Vocabulary not found. Please upload dataset first.'}), 404


@app.route('/start_training', methods=['POST'])
def start_training():
    """Frontend signals to begin data distribution. Resets ALL global state so that
    a second training session on Render starts cleanly (BUG-04 fix).
    """
    global round_status, data_shards, TOTAL_ROUNDS, LOCAL_EPOCHS
    global current_round, accuracy_history, epoch_reports, client_micro_states
    global submitted_weights, submitted_samples, registered_clients
    global client_registry, model_downloaded, _cached_val_loader

    data = request.get_json()
    num_clients = data.get('client_count')
    
    if 'rounds' in data:
        try:
            TOTAL_ROUNDS = int(data['rounds'])
        except ValueError:
            pass
            
    if 'epochs' in data:
        try:
            LOCAL_EPOCHS = int(data['epochs'])
        except ValueError:
            pass

    if not num_clients or num_clients < 1:
        return jsonify({'error': 'Invalid client_count'}), 400

    if not os.path.exists(uploaded_dataset_path):
        return jsonify({'error': 'No dataset uploaded yet'}), 400

    if len(registered_clients) < num_clients:
        return jsonify({
            'error': f'Only {len(registered_clients)} clients registered, but {num_clients} expected'
        }), 400

    # --- BUG-04: Full state reset for a clean second session ---
    with state_lock:
        current_round        = 1
        accuracy_history     = []
        epoch_reports        = {}
        client_micro_states  = {}
        submitted_weights    = {}
        submitted_samples    = {}
        model_downloaded     = False
        _cached_val_loader   = None
        # Keep registered_clients — they are still connected; just clear session flags
        client_registry      = {cid: {'data_received': False} for cid in registered_clients}
        data_shards          = {}
    print("[Coordinator] Global state reset for new training session.")

    try:
        # Divide dataset among registered clients
        print(f"\n[Coordinator] Dividing dataset for {num_clients} clients...")
        data_shards = data_distributor.divide_dataset(uploaded_dataset_path, num_clients)

        # Validate shards
        total_samples = sum(len(shard) for shard in data_shards.values())
        data_distributor.validate_shards(data_shards, total_samples)

        print(f"[Coordinator] Dataset divided successfully")
        for client_id, shard in data_shards.items():
            print(f"  {client_id}: {len(shard)} samples")

        round_status = 'data_distributing'
        return jsonify({'status': 'ok', 'shards_prepared': len(data_shards)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/get_data_shard', methods=['POST'])
def get_data_shard():
    """Send client's data shard as CSV file."""
    global round_status

    data = request.get_json()
    client_id = data.get('client_id')

    if not client_id:
        return jsonify({'error': 'client_id missing'}), 400

    # BUG-03 FIX: Allow shard download when status is 'data_distributing' OR 'active'
    # in round 1. The race: first client download triggers status→active before the
    # second client calls this endpoint (may be <1s later). Allowing 'active' here
    # fixes multi-client sessions. After round 1, data is already distributed.
    if round_status not in ('data_distributing', 'active'):
        return jsonify({'error': 'Data distribution not active'}), 400
    if current_round != 1:
        return jsonify({'error': 'Data already distributed (past round 1)'}), 400

    if client_id not in data_shards:
        return jsonify({'error': f'No shard for {client_id}'}), 400

    try:
        # Convert DataFrame to CSV in memory
        shard_df = data_shards[client_id]
        csv_data = shard_df.to_csv(index=False).encode('utf-8')

        # Mark as received
        if client_id in client_registry:
            client_registry[client_id]['data_received'] = True

        print(f"[Coordinator] Data shard sent to {client_id}")

        # Transition to 'active' immediately when the FIRST client requests a shard.
        with state_lock:
            if round_status == 'data_distributing':
                round_status = 'active'
                print(f"[Coordinator] Data served to {client_id}. Status → active. Training underway.")
                broadcast_event('sys_log', {
                    'message': f'Data shard sent to {client_id}. Training is now active.',
                    'level': 'info'
                })

        return csv_data, 200, {'Content-Disposition': 'attachment; filename="data_shard.csv"'}

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# --- NEW ENDPOINTS: Model Download, Leaderboard, Stats, SSE, Epoch Reports ---

@app.route('/download_model', methods=['GET'])
def download_model():
    """Download the trained global model. Triggers session cleanup after download."""
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not available'}), 503

    global model_downloaded
    model_downloaded = True
    print("[Coordinator] Model download requested — sending checkpoint.pt")

    # Schedule cleanup after a short delay to let the download complete
    def delayed_cleanup():
        import time
        time.sleep(5)  # Wait for download stream to finish
        cleanup_session_data()

    threading.Thread(target=delayed_cleanup, daemon=True).start()

    return send_file(MODEL_PATH, as_attachment=True,
                     download_name='net_neutral_trained_model.pt')


def cleanup_session_data():
    """Delete all session-specific data from Render to stay under 512MB.

    Keeps: credits database (persistent leaderboard), pretrained backup.
    Deletes: uploaded datasets, temp weight files, data shards, test splits.
    """
    print("[Coordinator] Starting session cleanup to free Render memory...")
    cleaned = []

    # 1. Delete uploaded dataset
    if os.path.exists(uploaded_dataset_path):
        os.remove(uploaded_dataset_path)
        cleaned.append('uploaded_dataset.csv')

    # 2. Delete test split
    test_csv = os.path.join(COORDINATOR_DIR, 'uploaded_test.csv')
    if os.path.exists(test_csv):
        os.remove(test_csv)
        cleaned.append('uploaded_test.csv')

    # 3. Delete all temp weight files
    for f in glob.glob(os.path.join(COORDINATOR_DIR, 'temp_*.pt')):
        os.remove(f)
        cleaned.append(os.path.basename(f))

    # 4. BUG-16 FIX: Only delete checkpoint.pt if the module-level restore confirmed it
    # works under WSGI.
    if os.path.exists(MODEL_PATH):
        PRETRAINED_BACKUP = os.path.join(COORDINATOR_DIR, 'checkpoint_pretrained_backup.pt')
        if os.path.exists(PRETRAINED_BACKUP) and _checkpoint_restored:
            # Restore baseline immediately so the next session starts clean
            import shutil
            shutil.copy2(PRETRAINED_BACKUP, MODEL_PATH)
            print("[Coordinator] Checkpoint restored to pretrained baseline after cleanup.")
            cleaned.append('checkpoint.pt (restored to baseline)')

    # 5. Clear in-memory caches
    data_shards.clear()
    epoch_reports.clear()

    print(f"[Coordinator] Session cleanup complete. Removed: {', '.join(cleaned) or 'nothing'}")
    broadcast_event('session_cleanup', {'files_removed': cleaned})


@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    """Public leaderboard — returns lifetime credits for all clients."""
    lb = credits.get_leaderboard_dicts()
    return jsonify({'leaderboard': lb})


@app.route('/stats', methods=['GET'])
def stats():
    """Public stats for landing page stats bar."""
    return jsonify(credits.get_stats())


@app.route('/report_epoch', methods=['POST'])
def report_epoch():
    """Receive per-epoch training metrics from a client.

    Called by client.py after each local training epoch to surface
    rich training data (loss, accuracy) in the web dashboard.
    """
    data = request.get_json()
    client_id = data.get('client_id')
    if not client_id:
        return jsonify({'error': 'client_id missing'}), 400

    epoch_data = {
        'epoch':    data.get('epoch', 0),
        'loss':     data.get('loss', 0.0),
        'accuracy': data.get('accuracy', 0.0),
        'samples':  data.get('samples', 0),
        'round':    data.get('round', current_round),
        'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
    }

    if client_id not in epoch_reports:
        epoch_reports[client_id] = []
    epoch_reports[client_id].append(epoch_data)

    # Broadcast via SSE for real-time dashboard updates
    broadcast_event('epoch_update', {
        'client_id': client_id,
        **epoch_data,
    })

    return jsonify({'status': 'ok'})


@app.route('/events')
def sse_stream():
    """Server-Sent Events endpoint for real-time updates.

    Browsers connect via EventSource and receive live training events
    without polling. Falls back to /status polling if SSE unavailable.
    """
    def stream():
        q = queue.Queue(maxsize=50)
        with sse_lock:
            sse_clients.append(q)
        try:
            # Send initial state
            init_data = json.dumps({
                'round': current_round,
                'round_status': round_status,
                'global_accuracy': global_accuracy,
                'accuracy_history': accuracy_history,
                'clients': list(registered_clients),
            })
            yield f"event: init\ndata: {init_data}\n\n"

            while True:
                try:
                    # BUG-18: 20s timeout keeps keepalive within Render's 30s idle
                    # connection close window (was 30s, which fired too late).
                    msg = q.get(timeout=20)
                    yield msg
                except queue.Empty:
                    # Send keepalive to prevent connection timeout
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(
        stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


@app.route('/')
def serve_frontend():
    """Serve frontend HTML."""
    project_root = os.path.dirname(os.path.dirname(COORDINATOR_DIR))
    frontend_path = os.path.join(project_root, 'frontend', 'index.html')
    if os.path.exists(frontend_path):
        return send_file(frontend_path)
    # In production if frontend not found, return API health instead of 404
    return jsonify({'status': 'ok', 'service': 'Net-Neutral AI Coordinator', 'version': '2.2'}), 200

# --- BUG-05: Module-level checkpoint restore ---
# This block runs whether the server is started via `python server.py` OR
# via gunicorn (which imports the module directly, never hitting __main__).
# Ensures every deploy starts from the clean pretrained baseline.
_checkpoint_restored = False

def _restore_checkpoint_baseline():
    """Restore checkpoint.pt from the pretrained backup at process startup.
    Called at module import time so gunicorn workers also get a clean baseline.
    """
    global _checkpoint_restored
    PRETRAINED_BACKUP = os.path.join(COORDINATOR_DIR, 'checkpoint_pretrained_backup.pt')
    if os.path.exists(PRETRAINED_BACKUP):
        import shutil
        shutil.copy2(PRETRAINED_BACKUP, MODEL_PATH)
        _checkpoint_restored = True
        print("[Coordinator] [OK] Checkpoint reset to pretrained baseline")
    elif not os.path.exists(MODEL_PATH):
        print(f"[Coordinator] ⚠  No checkpoint found. Generating dummy for startup.")
        torch.save({"dummy": torch.tensor([1.0])}, MODEL_PATH)
    else:
        print("[Coordinator] ⚠  No backup found — using existing checkpoint.pt")

# Run immediately at import (covers both direct run and gunicorn)
_restore_checkpoint_baseline()


if __name__ == '__main__':
    # Bind to PORT environment variable if available (e.g. on Render), default to 5000
    port = int(os.environ.get("PORT", 5000))
    print("[Coordinator] Starting Net-Neutral AI Coordinator...")
    print(f"[Coordinator] Server running on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
