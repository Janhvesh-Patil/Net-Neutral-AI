import os
import datetime
import torch
from flask import Flask, request, jsonify, send_file

# Import internal modules [cite: 2, 266-267]
import fedavg
import credits
import evaluate

app = Flask(__name__)

# --- PATH CONFIGURATION ---
# Ensures the server always looks in the exact folder where server.py lives
COORDINATOR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(COORDINATOR_DIR, 'global_model.pt')

# --- Global State Machine ---
current_round = 1
TOTAL_ROUNDS = 5  # Configured per TRD spec [cite: 2, 303-305]
registered_clients = set()
round_status = 'active'
global_accuracy = 0.0
round_start_time = datetime.datetime.now()

# Dictionaries to track submissions
submitted_weights = {}  
submitted_samples = {}  

# --- Evaluation Wrapper ---
def run_evaluation_from_path(model_path: str, round_number: int) -> float:
    """
    Wrapper to load the model and data, then call the CS team's evaluate function.
    Because Machine A runs both the server and Client A, it has access to the client folder [cite: 2, 237-238].
    """
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
    
    # 3. Run the CS teammate's core evaluate function
    print("\n[Coordinator] Running global evaluation...")
    result = evaluate.evaluate(model, val_loader)
    
    # 4. Print the formatted results to the terminal
    prev_accuracy = credits._get_previous_accuracy(round_number, credits.DB_PATH) if round_number > 1 else 0.0
    print(evaluate.format_eval_result(result, round_num=round_number, total_rounds=TOTAL_ROUNDS, prev_accuracy=prev_accuracy))
    
    return result.accuracy

# --- Core Aggregation Logic ---
def check_round_completion():
    global current_round, round_status, global_accuracy, round_start_time
    
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

        # 6. Step the state machine forward [cite: 1, 121-125]
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
    """Called once per client at startup [cite: 1, 98-99]."""
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