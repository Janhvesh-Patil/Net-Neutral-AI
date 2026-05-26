import os
import datetime
import torch
from flask import Flask, request, jsonify, send_file

# Import the custom modules provided by the CS team [cite: 2, 266-267]
import fedavg
import credits

app = Flask(__name__)

# --- Global State Machine ---
# Tracks the state of the federated learning loop [cite: 1, 121-125]
current_round = 1
TOTAL_ROUNDS = 5  # Configured per TRD spec [cite: 2, 303-305]
registered_clients = set()
round_status = 'active'
global_accuracy = 0.0
round_start_time = datetime.datetime.now()

# Dictionaries to track submissions before triggering FedAvg
submitted_weights = {}  # client_id -> file_path
submitted_samples = {}  # client_id -> samples_trained

# --- Core Aggregation Logic ---
def check_round_completion():
    """Checks if all clients have submitted. If so, runs FedAvg and logs the round."""
    global current_round, round_status, global_accuracy, round_start_time
    
    if len(submitted_weights) >= len(registered_clients) and len(registered_clients) > 0:
        round_status = 'aggregating'
        print(f"\n--- All clients submitted for round {current_round}. Running FedAvg ---")
        
        # 1. Load all submitted client weights
        client_weights = {}
        for cid, fpath in submitted_weights.items():
            state_dict, err = fedavg.load_client_weights(fpath, cid)
            if not err:
                client_weights[cid] = state_dict
            else:
                print(f"[Coordinator] ⚠ Error loading {cid}: {err}")
        
        try:
            # 2. Run the math engine (Federated Averaging)
            result = fedavg.federated_average(client_weights, client_samples=submitted_samples)
            
            # 3. Save the new, averaged global model to disk
            fedavg.save_global_model(result.global_state_dict, 'global_model.pt')
            
            # 4. Evaluate (Mocked for demo until ML team integrates the validation set) [cite: 2, 338-341]
            global_accuracy += 12.5 
            
            # 5. The Accountant: Log the round to SQLite and print leaderboard
            credits.log_round(current_round, round_start_time, result.clients_included, global_accuracy)
            board = credits.get_leaderboard()
            
            if current_round >= TOTAL_ROUNDS:
                print(credits.format_final_leaderboard(board))
            else:
                print(credits.format_leaderboard(board, current_round))
                
        except fedavg.FedAvgError as e:
            print(f"[Coordinator] ⚠ FedAvg failed: {e}")
        
        # 6. Clean up temporary weight files
        for fpath in submitted_weights.values():
            if os.path.exists(fpath):
                os.remove(fpath)
        submitted_weights.clear()
        submitted_samples.clear()

        # 7. Step the state machine forward
        if current_round >= TOTAL_ROUNDS:
            round_status = 'done'
            print("\n[Coordinator] Training complete! Final leaderboard ready.")
        else:
            current_round += 1
            round_status = 'active'
            round_start_time = datetime.datetime.now()
            print(f"\n[Coordinator] Starting Round {current_round}")

# --- API Endpoints ---

@app.route('/register', methods=['POST'])
def register():
    """Called once per client at startup [cite: 1, 98-100]."""
    data = request.get_json()
    client_id = data.get('client_id')
    
    if not client_id:
        return jsonify({'error': 'client_id missing'}), 400
        
    registered_clients.add(client_id)
    print(f"[Coordinator] Node Registered: {client_id}")
    
    return jsonify({'status': 'ok', 'round': current_round})

@app.route('/model', methods=['GET'])
def get_model():
    """Serves the binary .pt global model file [cite: 1, 107-108]."""
    return send_file('global_model.pt', as_attachment=True)

@app.route('/submit', methods=['POST'])
def submit():
    """Receives local weights and metadata via multipart form-data [cite: 1, 113-115]."""
    global round_status
    
    client_id = request.form['client_id']
    samples_trained = int(request.form['samples_trained'])
    time_seconds = float(request.form['time_seconds'])
    weights_file = request.files['weights']
    
    # Save the binary weight file temporarily [cite: 1, 117-118]
    save_path = f"temp_{client_id}_round{current_round}.pt"
    weights_file.save(save_path)
    
    submitted_weights[client_id] = save_path
    submitted_samples[client_id] = samples_trained
    
    # The Accountant: Log this submission directly to SQLite
    points_earned = credits.log_credit(client_id, current_round, samples_trained, time_seconds)
    
    # Check if this was the last client needed to trigger FedAvg
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

if __name__ == '__main__':
    # Initialize the SQLite database automatically
    credits.init_db()
    
    # Generate a safe dummy global_model.pt so the first GET /model request doesn't crash
    if not os.path.exists('global_model.pt'):
        print("[Coordinator] global_model.pt not found. Generating a safe dummy checkpoint for startup.")
        torch.save({"dummy": torch.tensor([1.0])}, 'global_model.pt')

    # CRITICAL DEMO-SAVER: host='0.0.0.0' binds to the WiFi adapter.
    print("[Coordinator] Starting Net-Neutral AI Coordinator...")
    print("[Coordinator] Server running on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)