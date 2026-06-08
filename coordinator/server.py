import os
import datetime
import torch
from flask import Flask, request, jsonify, send_file

# Import internal modules
import fedavg
import credits
import evaluate
import data_distributor

app = Flask(__name__, static_folder='../frontend', static_url_path='/')

# --- PATH CONFIGURATION ---
# Ensures the server always looks in the exact folder where server.py lives
COORDINATOR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(COORDINATOR_DIR, 'checkpoint.pt')

# --- Global State Machine ---
current_round = 1
TOTAL_ROUNDS = 5
registered_clients = set()
round_status = 'waiting_for_clients'
global_accuracy = 0.0
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

# --- Evaluation Wrapper ---
def run_evaluation_from_path(model_path: str, round_number: int) -> float:
    import sys
    # Ensure Python can find the client folder
    project_root = os.path.dirname(COORDINATOR_DIR)
    sys.path.insert(0, project_root)
    
    from client.model import TransformerClassifier
    from client.data import setup_data, get_validation_dataloader
    
    # 1. Load the validation dataset
    _, _, test_texts, test_labels, vocab = setup_data(data_dir=os.path.join(project_root, "data"), save_vocab=False)
    val_loader = get_validation_dataloader(test_texts, test_labels, vocab)
    
    # 2. Initialize an empty model and load the weights from the file
    model = TransformerClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    print("\n[Coordinator] Running global evaluation...")
    result = evaluate.evaluate(model, val_loader)
    
    # 4. Print the formatted results to the terminal
    prev_accuracy = credits._get_previous_accuracy(round_number, credits.DB_PATH) if round_number > 1 else 0.0
    print(evaluate.format_eval_result(result, round_num=round_number, total_rounds=TOTAL_ROUNDS, prev_accuracy=prev_accuracy))
    
    return result.accuracy

# --- Core Aggregation Logic ---
def check_round_completion():
    global current_round, round_status, global_accuracy, round_start_time

    # Don't check for completion if waiting for clients or distributing data
    if round_status in ['waiting_for_clients', 'data_distributing']:
        return

    if len(submitted_weights) >= len(registered_clients) > 0:
        round_status = 'aggregating'
        print(f"\n--- All clients submitted for round {current_round}. Running FedAvg ---")

        client_weights = {}
        for cid, fpath in submitted_weights.items():
            state_dict, err = fedavg.load_client_weights(fpath, cid)
            if not err:
                client_weights[cid] = state_dict
            else:
                print(f"[Coordinator] ⚠ Error loading {cid}: {err}")

        try:
            # 1. Run FedAvg
            result = fedavg.federated_average(client_weights, client_samples=submitted_samples)

            # 2. Save using the absolute MODEL_PATH
            fedavg.save_global_model(result.global_state_dict, MODEL_PATH)

            # 3. Evaluate the new global model
            global_accuracy = run_evaluation_from_path(MODEL_PATH, current_round)

            # 4. Log the round and print leaderboard
            credits.log_round(current_round, round_start_time, result.clients_included, global_accuracy)
            board = credits.get_leaderboard()

            if current_round >= TOTAL_ROUNDS:
                print(credits.format_final_leaderboard(board))
            else:
                print(credits.format_leaderboard(board, current_round))

        except fedavg.FedAvgError as e:
            print(f"[Coordinator] ⚠ FedAvg failed: {e}")
        except Exception as e:
            print(f"[Coordinator] ⚠ Evaluation failed: {e}")

        # 5. Clean up temporary weight files (FIXED os.remove)
        for fpath in list(submitted_weights.values()):
            if os.path.exists(fpath):
                os.remove(fpath)
        submitted_weights.clear()
        submitted_samples.clear()

        # 6. Step the state machine forward
        if current_round >= TOTAL_ROUNDS:
            round_status = 'done'
            print("\n[Coordinator] Training complete! Final leaderboard ready.")

            # Sync credits to central Supabase database
            try:
                import supabase_sync
                supabase_sync.sync_credits_to_cloud()
            except Exception as e:
                print(f"[Coordinator] Cloud sync failed (non-fatal): {e}")
        else:
            current_round += 1
            # FIX 1.1: After round 1, data is already distributed — go directly
            # to 'active' so check_round_completion() doesn't return early.
            # Only the first /start_training call uses 'data_distributing'.
            round_status = 'active'
            round_start_time = datetime.datetime.now()
            print(f"\n[Coordinator] Ready for Round {current_round}")

# --- API Endpoints ---

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

    return jsonify({'status': 'ok', 'round': current_round})

@app.route('/model', methods=['GET'])
def get_model():
    """Serves the binary .pt global model file [cite: 1, 107-108]."""
    return send_file(MODEL_PATH, as_attachment=True)

@app.route('/submit', methods=['POST'])
def submit():
    """Receives local weights and metadata via multipart form-data [cite: 1, 113-115]."""
    global round_status
    
    client_id = request.form['client_id']
    samples_trained = int(request.form['samples_trained'])
    time_seconds = float(request.form['time_seconds'])
    weights_file = request.files['weights']
    
    # Save the binary weight file temporarily in the absolute dir [cite: 1, 117-118]
    save_path = os.path.join(COORDINATOR_DIR, f"temp_{client_id}_round{current_round}.pt")
    weights_file.save(save_path)
    
    submitted_weights[client_id] = save_path
    submitted_samples[client_id] = samples_trained
    
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
    """Polled every 5 seconds by clients to check state machine [cite: 1, 120-121]."""
    return jsonify({
        'round': current_round,
        'round_status': round_status,
        'active_clients': list(registered_clients)
    })

@app.route('/results', methods=['GET'])
def results():
    """Returns final dashboard stats."""
    return jsonify({
        'final_accuracy': global_accuracy,
        'total_rounds_completed': current_round if round_status == 'done' else current_round - 1
    })


@app.route('/api/config', methods=['GET'])
def api_config():
    """Returns server configuration for the frontend."""
    return jsonify({
        'max_clients': 3,
        'total_rounds': TOTAL_ROUNDS,
        'current_round': current_round,
        'round_status': round_status,
    })


@app.route('/api/client_status/<client_id>', methods=['GET'])
def api_client_status(client_id):
    """Returns individual client's credits and round participation."""
    client = credits.get_client(client_id)
    if client is None:
        return jsonify({'error': f'Client {client_id} not found'}), 404

    total_credits = credits.get_client_total_credits(client_id)
    submitted = credits.get_submitted_clients(current_round)

    return jsonify({
        'client_id': client_id,
        'ip_address': client.ip_address,
        'registered_at': client.registered_at,
        'total_credits': total_credits,
        'current_round': current_round,
        'round_status': round_status,
        'total_rounds': TOTAL_ROUNDS,
        'has_submitted_this_round': client_id in submitted,
        'global_accuracy': global_accuracy,
    })


# --- NEW ENDPOINTS FOR DATA DISTRIBUTION ---

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """Accept CSV dataset upload from frontend."""
    global round_status

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be CSV format'}), 400

    try:
        # Save uploaded file
        file.save(uploaded_dataset_path)
        print(f"[Coordinator] Dataset uploaded: {uploaded_dataset_path}")

        # Validate the dataset
        df = data_distributor.load_and_validate_csv(uploaded_dataset_path)
        num_rows = len(df)

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


@app.route('/start_training', methods=['POST'])
def start_training():
    """Frontend signals to begin data distribution."""
    global round_status, data_shards

    data = request.get_json()
    num_clients = data.get('client_count')

    if not num_clients or num_clients < 1:
        return jsonify({'error': 'Invalid client_count'}), 400

    if not os.path.exists(uploaded_dataset_path):
        return jsonify({'error': 'No dataset uploaded yet'}), 400

    if len(registered_clients) < num_clients:
        return jsonify({
            'error': f'Only {len(registered_clients)} clients registered, but {num_clients} expected'
        }), 400

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

    # Only distribute data in first round (data_distributing state)
    if round_status != 'data_distributing':
        return jsonify({'error': 'Data distribution not active'}), 400

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

        # Check if all clients have received data
        all_received = all(
            client_registry.get(cid, {}).get('data_received', False)
            for cid in data_shards.keys()
        )

        if all_received and round_status == 'data_distributing':
            round_status = 'active'
            print(f"[Coordinator] All clients received data. Starting training...")

        return csv_data, 200, {'Content-Disposition': 'attachment; filename="data_shard.csv"'}

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/')
def serve_frontend():
    """Serve frontend HTML."""
    frontend_path = os.path.join(os.path.dirname(COORDINATOR_DIR), 'frontend', 'index.html')
    if os.path.exists(frontend_path):
        return send_file(frontend_path)
    return jsonify({'error': 'Frontend not found'}), 404

if __name__ == '__main__':
    # Initialize the SQLite database automatically
    credits.init_db()
    
    # Use absolute MODEL_PATH for the startup safety check
    if not os.path.exists(MODEL_PATH):
        print(f"[Coordinator] {MODEL_PATH} not found. Generating a safe dummy checkpoint for startup.")
        # Create a tiny dummy state dict
        torch.save({"dummy": torch.tensor([1.0])}, MODEL_PATH)

    # CRITICAL DEMO-SAVER: host='0.0.0.0' binds to the WiFi adapter.
    print("[Coordinator] Starting Net-Neutral AI Coordinator...")
    print("[Coordinator] Server running on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)