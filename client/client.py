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
from shared import ip_utils
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

    # Get client IP address (NEW)
    try:
        client_ip = ip_utils.get_local_ip()
    except Exception:
        client_ip = "unknown"

    payload = {
        "client_id": client_id,
        "ip_address": client_ip
    }

    for attempt in range(1, config.REGISTER_RETRY_ATTEMPTS + 1):
        try:
            print_status(f"Registering with coordinator from {client_ip} (attempt {attempt}/{config.REGISTER_RETRY_ATTEMPTS})...")
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            data = response.json()
            current_round = data.get('round', 1)

            print_status(f"[OK] Registered successfully. Current round: {current_round}")
            return current_round

        except requests.exceptions.RequestException as e:
            print_status(f"[ERROR] Registration failed: {e}")
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

        print_status(f"[OK] Model downloaded to {save_path}")

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
        print_status(f"[OK] Submission successful. Credits earned: {result.get('credits', 0)}")

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
                print_status("[OK] All rounds complete!")
                return status

            if status.get('round') > current_round:
                print_status(f"[OK] Round {status.get('round')} started!")
                return status

            if elapsed % 10 == 0:
                print_status(f"Waiting for round {current_round + 1}... ({elapsed}s elapsed)")

            time.sleep(config.POLL_INTERVAL_SECS)

        except requests.exceptions.RequestException as e:
            print_status(f"[WARNING] Status poll failed: {e}. Retrying...")
            time.sleep(config.POLL_INTERVAL_SECS)


def download_data_shard(client_id: str, save_path: str, max_retries: int = 3) -> None:
    """
    Download client's data shard from coordinator.
    Only used in first round; subsequent rounds use cached data.
    """
    url = f"{config.BASE_URL}/get_data_shard"
    payload = {"client_id": client_id}

    for attempt in range(1, max_retries + 1):
        try:
            print_status(f"Downloading data shard (attempt {attempt}/{max_retries})...")
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'wb') as f:
                f.write(response.content)

            print_status(f"[OK] Data shard saved: {save_path}")
            return

        except Exception as e:
            if attempt < max_retries:
                print_status(f"[WARNING] Download failed (attempt {attempt}/{max_retries}): {e}")
                time.sleep(5)
            else:
                raise RuntimeError(f"Failed to download data shard after {max_retries} attempts: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLIENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_client(client_id: str, data_dir: str, vocab_path: str) -> None:

    print_banner(client_id)

    # ── Step 1: Register with coordinator ─────────────────────────────────────
    current_round = register_with_coordinator(client_id)

    # ── Step 2: Download data shard if not cached (NEW) ─────────────────────────
    data_shard_path = os.path.join(config.LOCAL_DATA_DIR, config.DATA_SHARD_FILENAME.format(client_id=client_id))

    if not os.path.exists(data_shard_path):
        # FIX 1.4: Wait for coordinator to enter 'data_distributing' or 'active'
        # before attempting to download the shard.  Without this, the client
        # would crash because /get_data_shard returns 400 when distribution
        # hasn't been triggered yet via /start_training.
        print_status("Waiting for coordinator to start data distribution...")
        wait_start = time.time()
        while True:
            try:
                resp = requests.get(f"{config.BASE_URL}/status", timeout=5)
                resp.raise_for_status()
                st = resp.json().get('round_status', '')
                if st in ('data_distributing', 'active'):
                    print_status(f"[OK] Coordinator status: {st} — downloading shard")
                    break
                elapsed = int(time.time() - wait_start)
                if elapsed > config.WAIT_FOR_DATA_TIMEOUT_SECS:
                    raise RuntimeError(
                        f"Timed out after {config.WAIT_FOR_DATA_TIMEOUT_SECS}s waiting "
                        f"for data distribution. Current status: {st}"
                    )
                if elapsed % 10 == 0:
                    print_status(f"Coordinator status: {st} — waiting... ({elapsed}s)")
                time.sleep(config.POLL_INTERVAL_SECS)
            except requests.exceptions.RequestException as e:
                print_status(f"[WARNING] Status poll failed: {e}. Retrying...")
                time.sleep(config.POLL_INTERVAL_SECS)

        print_status(f"Data shard not found locally. Downloading from coordinator...")
        try:
            download_data_shard(client_id, data_shard_path)
        except Exception as e:
            print_status(f"[ERROR] {e}")
            raise
    else:
        print_status(f"[OK] Using cached data shard: {data_shard_path}")

    # ── Step 3: Load local data shard ─────────────────────────────────────────
    print_status("Loading data shard...")
    train_texts, train_labels, _, _, vocab = setup_data(
        data_dir=data_dir,
        vocab_path=vocab_path,
        save_vocab=False,
        local_shard_path=data_shard_path
    )

    # FIX 1.2: When local_shard_path is used the CSV already contains only
    # this client's data.  get_client_dataloader() would slice by the
    # hardcoded SHARD_RANGES and return empty/wrong data for client_B/C.
    if os.path.exists(data_shard_path):
        from data import get_full_dataloader
        dataloader = get_full_dataloader(
            texts=train_texts,
            labels=train_labels,
            vocab=vocab,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
        )
    else:
        dataloader = get_client_dataloader(
            client_id=client_id,
            train_texts=train_texts,
            train_labels=train_labels,
            vocab=vocab,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
        )

    print_status(f"[OK] Data loaded. Shard size: {len(dataloader.dataset):,} samples")

    # ── Step 4: Training loop ────────────────────────────────────────────────────
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
            print_status("[OK] Global model loaded")

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
            print_status(f"[OK] Weights saved to {weights_path}")

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
                print_status("[OK] All rounds complete!")
                break

            status = poll_for_next_round(round_num)

            if status.get('round_status') == 'done':
                print_status("[OK] Training session complete!")
                break

        except Exception as e:
            print_status(f"[ERROR] Error in round {round_num}: {e}")
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
