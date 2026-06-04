import os
import sys
import time
import argparse
import tempfile
import requests
import torch

# Add parent directory to path to import shared config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import config
from model import TransformerClassifier
from train import train_one_round, save_weights, load_weights
from data import setup_data, get_client_dataloader


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(client_id: str) -> None:
    print("\n" + "=" * 60)
    print(f"  Net-Neutral AI | {client_id.upper()}")
    print("=" * 60)
    print(f"  Coordinator: {config.BASE_URL}")
    print(f"  Total Rounds: {config.TOTAL_ROUNDS}")
    print(f"  Local Epochs: {config.LOCAL_EPOCHS}")
    print("=" * 60 + "\n")


def print_status(message: str, prefix: str = "[Client]") -> None:
    print(f"{prefix} {message}")


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK COMMUNICATION
# ─────────────────────────────────────────────────────────────────────────────

def register_with_coordinator(client_id: str) -> int:

    url = f"{config.BASE_URL}/register"
    payload = {"client_id": client_id}
    
    for attempt in range(1, config.REGISTER_RETRY_ATTEMPTS + 1):
        try:
            print_status(f"Registering with coordinator (attempt {attempt}/{config.REGISTER_RETRY_ATTEMPTS})...")
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current_round = data.get('round', 1)
            
            print_status(f"✓ Registered successfully. Current round: {current_round}")
            return current_round
            
        except requests.exceptions.RequestException as e:
            print_status(f"✗ Registration failed: {e}")
            if attempt < config.REGISTER_RETRY_ATTEMPTS:
                print_status(f"Retrying in {config.REGISTER_RETRY_DELAY} seconds...")
                time.sleep(config.REGISTER_RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"Failed to register after {config.REGISTER_RETRY_ATTEMPTS} attempts. "
                    f"Check that coordinator is running at {config.BASE_URL} and "
                    f"COORDINATOR_IP is correct in shared/config.py"
                )


def download_global_model(save_path: str) -> None:

    url = f"{config.BASE_URL}/model"
    
    try:
        print_status("Downloading global model...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print_status(f"✓ Model downloaded to {save_path}")
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download global model: {e}")


def submit_weights(
    client_id: str,
    weights_path: str,
    samples_trained: int,
    time_seconds: float,
) -> dict:

    url = f"{config.BASE_URL}/submit"
    
    try:
        print_status("Submitting weights to coordinator...")
        
        with open(weights_path, 'rb') as f:
            files = {'weights': f}
            data = {
                'client_id': client_id,
                'samples_trained': samples_trained,
                'time_seconds': time_seconds,
            }
            
            response = requests.post(url, files=files, data=data, timeout=60)
            response.raise_for_status()
        
        result = response.json()
        print_status(f"✓ Submission successful. Credits earned: {result.get('credits', 0)}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to submit weights: {e}")


def poll_for_next_round(current_round: int) -> dict:

    url = f"{config.BASE_URL}/status"
    wait_start = time.time()
    
    print_status(f"Waiting for round {current_round + 1}...")
    
    while True:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            status = response.json()
            
            elapsed = int(time.time() - wait_start)

            if status.get('round_status') == 'done':
                print_status("✓ All rounds complete!")
                return status

            if status.get('round') > current_round:
                print_status(f"✓ Round {status.get('round')} started!")
                return status

            if elapsed % 10 == 0:
                print_status(f"Waiting for round {current_round + 1}... ({elapsed}s elapsed)")
            
            time.sleep(config.POLL_INTERVAL_SECS)
            
        except requests.exceptions.RequestException as e:
            print_status(f"⚠ Status poll failed: {e}. Retrying...")
            time.sleep(config.POLL_INTERVAL_SECS)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLIENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_client(client_id: str, data_dir: str, vocab_path: str) -> None:

    print_banner(client_id)
    
    # ── Step 1: Register with coordinator ─────────────────────────────────────
    current_round = register_with_coordinator(client_id)
    
    # ── Step 2: Load local data shard ─────────────────────────────────────────
    print_status("Loading local data shard...")
    train_texts, train_labels, _, _, vocab = setup_data(
        data_dir=data_dir,
        vocab_path=vocab_path,
        save_vocab=False,
    )
    
    dataloader = get_client_dataloader(
        client_id=client_id,
        train_texts=train_texts,
        train_labels=train_labels,
        vocab=vocab,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    
    print_status(f"✓ Data loaded. Shard size: {len(dataloader.dataset):,} samples")
    
    # ── Step 3: Training loop ─────────────────────────────────────────────────
    for round_num in range(current_round, config.TOTAL_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num} / {config.TOTAL_ROUNDS}")
        print(f"{'='*60}\n")
        
        # Download global model
        model_temp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
        model_path = model_temp.name
        model_temp.close()
        
        try:
            download_global_model(model_path)

            model = TransformerClassifier()
            model = load_weights(model_path, model)
            print_status("✓ Global model loaded")

            state_dict, samples_trained, time_seconds, final_loss = train_one_round(
                model=model,
                dataloader=dataloader,
                client_id=client_id,
                round_num=round_num,
                total_rounds=config.TOTAL_ROUNDS,
                epochs=config.LOCAL_EPOCHS,
                lr=config.LEARNING_RATE,
            )

            weights_temp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
            weights_path = weights_temp.name
            weights_temp.close()
            
            save_weights(state_dict, weights_path)
            print_status(f"✓ Weights saved to {weights_path}")

            result = submit_weights(
                client_id=client_id,
                weights_path=weights_path,
                samples_trained=samples_trained,
                time_seconds=time_seconds,
            )

            print(f"\n{'─'*60}")
            print(f"  Round {round_num} Summary")
            print(f"{'─'*60}")
            print(f"  Credits earned    : {result.get('credits', 0)}")
            print(f"  Global accuracy   : {result.get('global_acc', 0.0):.1%}")
            print(f"  Samples trained   : {samples_trained:,}")
            print(f"  Training time     : {time_seconds:.1f}s")
            print(f"{'─'*60}\n")

            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(weights_path):
                os.unlink(weights_path)

            if round_num >= config.TOTAL_ROUNDS:
                print_status("✓ All rounds complete!")
                break

            status = poll_for_next_round(round_num)
            
            if status.get('round_status') == 'done':
                print_status("✓ Training session complete!")
                break
                
        except Exception as e:
            print_status(f"✗ Error in round {round_num}: {e}")
            if os.path.exists(model_path):
                os.unlink(model_path)
            if 'weights_path' in locals() and os.path.exists(weights_path):
                os.unlink(weights_path)
            raise
    
    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {client_id.upper()} | TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total rounds completed: {config.TOTAL_ROUNDS}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Net-Neutral AI Federated Client')
    parser.add_argument(
        '--client_id',
        type=str,
        default=config.CLIENT_ID,
        help=f'Client identifier (default: {config.CLIENT_ID})'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Path to data directory (default: ../data relative to client folder)'
    )
    parser.add_argument(
        '--vocab_path',
        type=str,
        default=None,
        help='Path to vocabulary JSON file (default: ../data/vocab.json)'
    )
    
    args = parser.parse_args()

    client_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(client_dir)
    
    data_dir = args.data_dir or os.path.join(project_root, 'data')
    vocab_path = args.vocab_path or os.path.join(data_dir, 'vocab.json')

    valid_clients = ['client_A', 'client_B', 'client_C']
    if args.client_id not in valid_clients:
        print(f"Warning: client_id '{args.client_id}' not in standard list {valid_clients}")
        print("Proceeding anyway...")

    # Run client
    try:
        run_client(args.client_id, data_dir, vocab_path)
    except KeyboardInterrupt:
        print("\n\n[Client] Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[Client] Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
