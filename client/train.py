"""
train.py — Net-Neutral AI
Local training loop for each federated client.

Spec reference: TRD Section 6.2, App Flow Section 5.3

Responsibilities:
    - Load global model weights received from coordinator
    - Train on local data shard for LOCAL_EPOCHS epochs
    - Return updated state_dict + metadata (samples, time, loss)
    - Print formatted terminal output after each epoch and on completion

Returns to client.py:
    state_dict      : updated model weights (sent to coordinator via /submit)
    samples_trained : total samples processed (used for credit calculation)
    time_seconds    : wall-clock training time
    final_loss      : average loss of the last epoch
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any

# ── Local imports ─────────────────────────────────────────────────────────────
from model import TransformerClassifier


# ── Constants ────────────────────────────────────────────────────────────────
LOCAL_EPOCHS  = 2
LEARNING_RATE = 1e-3
PRINT_WIDTH   = 60     # terminal output width


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _line(char: str = "─") -> str:
    return char * PRINT_WIDTH

def _header(title: str) -> None:
    print()
    print(_line("═"))
    print(f"  {title}")
    print(_line("═"))

def _section(title: str) -> None:
    print(_line("─"))
    print(f"  {title}")
    print(_line("─"))

def _row(label: str, value: str) -> None:
    print(f"  {label:<28}{value}")

def _blank() -> None:
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CORE TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_one_round(
    model:       TransformerClassifier,
    dataloader:  DataLoader,
    client_id:   str,
    round_num:   int,
    total_rounds: int,
    epochs:      int = LOCAL_EPOCHS,
    lr:          float = LEARNING_RATE,
) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
    """
    Run one federated training round — train for `epochs` epochs on local data.

    Args:
        model        : TransformerClassifier loaded with global weights
        dataloader   : client's local DataLoader (5,000 samples)
        client_id    : e.g. 'client_A' — used in terminal output
        round_num    : current round number (1-indexed)
        total_rounds : total rounds in session (e.g. 5)
        epochs       : number of local epochs per round
        lr           : learning rate for Adam optimiser

    Returns:
        state_dict      : updated model weights after local training
        samples_trained : total samples processed this round
        time_seconds    : wall-clock seconds for the full round
        final_loss      : average CrossEntropyLoss of the last epoch
    """

    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # ── Print round header ────────────────────────────────────────────────────
    _header(f"Net-Neutral AI   |   {client_id.upper()}   |   Round {round_num} / {total_rounds}")
    _row("Device", str(device).upper())

    if device.type == "cpu":
        print()
        print("  ⚠  Running on CPU — training will take 8–12 minutes.")
        print("     Shift to Colab/GPU for faster iteration.")

    _row("Epochs this round", str(epochs))
    _row("Samples in shard",  f"{len(dataloader.dataset):,}")
    _row("Batch size",        str(dataloader.batch_size))
    _row("Learning rate",     str(lr))
    _blank()

    # ── Optimiser and loss ────────────────────────────────────────────────────
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    samples_trained = 0
    final_loss      = 0.0
    start_time      = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start    = time.time()
        epoch_loss     = 0.0
        correct        = 0
        total          = 0
        num_batches    = 0

        for batch_ids, batch_labels in dataloader:
            # Move to device
            batch_ids    = batch_ids.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            optimiser.zero_grad()
            logits = model(batch_ids)               # (batch, 2)
            loss   = criterion(logits, batch_labels)

            # Backward pass
            loss.backward()
            # Gradient clipping — prevents exploding gradients on tricky batches
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            # Accumulate metrics
            epoch_loss  += loss.item()
            preds        = logits.argmax(dim=1)
            correct     += (preds == batch_labels).sum().item()
            total       += batch_labels.size(0)
            num_batches += 1

        # ── Per-epoch output ──────────────────────────────────────────────────
        avg_loss     = epoch_loss / num_batches
        accuracy     = correct / total * 100
        epoch_time   = time.time() - epoch_start
        final_loss   = avg_loss
        samples_trained += total

        _section(f"Epoch {epoch} / {epochs}")
        _row("Avg loss",          f"{avg_loss:.4f}")
        _row("Local accuracy",    f"{accuracy:.2f}%")
        _row("Samples processed", f"{total:,}")
        _row("Epoch time",        f"{epoch_time:.1f}s")

    # ── Round complete ────────────────────────────────────────────────────────
    time_seconds = time.time() - start_time

    _line_str = _line("─")
    print(_line_str)
    _section("Round Training Complete")
    _row("Total samples trained", f"{samples_trained:,}")
    _row("Total time",            f"{time_seconds:.1f}s")
    _row("Final loss",            f"{final_loss:.4f}")
    _row("Credits formula",       f"floor({samples_trained} / 5) = {samples_trained // 5} pts")
    print(_line("═"))
    _blank()

    # ── Return updated weights ────────────────────────────────────────────────
    # Move model back to CPU before extracting state_dict.
    # Reason: coordinator runs on CPU (IT's machine).
    # Sending GPU tensors causes silent errors on the receiving end.
    model.cpu()
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    return state_dict, samples_trained, time_seconds, final_loss


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT SERIALISATION HELPERS
# (called by client.py before POSTing to /submit)
# ─────────────────────────────────────────────────────────────────────────────

def save_weights(state_dict: Dict[str, torch.Tensor], path: str) -> None:
    """
    Save a state_dict to a .pt file for network transfer.

    Args:
        state_dict : model.state_dict() output
        path       : file path to save to (e.g. '/tmp/client_A_round1.pt')
    """
    torch.save(state_dict, path)


def load_weights(path: str, model: TransformerClassifier) -> TransformerClassifier:
    """
    Load a .pt file into a model instance.
    Always loads to CPU regardless of where the file was saved.

    Args:
        path  : path to .pt file
        model : TransformerClassifier instance to load weights into

    Returns:
        model with updated weights
    """
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    # Allow running from project root or client/ subfolder
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data import setup_data, get_client_dataloader

    print("=" * PRINT_WIDTH)
    print("  train.py sanity check")
    print("  Uses client_A shard — 1 epoch only for speed")
    print("=" * PRINT_WIDTH)

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir   = sys.argv[1] if len(sys.argv) > 1 else None
    vocab_path = os.path.join(data_dir, "vocab.json") if data_dir else None

    print("\n[train.py] Loading data...")
    train_texts, train_labels, _, _, vocab = setup_data(
        data_dir   = data_dir,
        vocab_path = vocab_path,
        save_vocab = False,   # vocab already built by data.py
    )

    dataloader = get_client_dataloader(
        client_id    = "client_A",
        train_texts  = train_texts,
        train_labels = train_labels,
        vocab        = vocab,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    print("[train.py] Building model...")
    model = TransformerClassifier()

    # ── Run one training round (1 epoch for speed in sanity check) ────────────
    state_dict, samples, time_secs, loss = train_one_round(
        model        = model,
        dataloader   = dataloader,
        client_id    = "client_A",
        round_num    = 1,
        total_rounds = 5,
        epochs       = 1,       # 1 epoch only for sanity check
        lr           = LEARNING_RATE,
    )

    # ── Verify outputs ────────────────────────────────────────────────────────
    print("[train.py] Verifying outputs...")
    assert isinstance(state_dict, dict),        "state_dict must be a dict"
    assert len(state_dict) > 0,                 "state_dict must not be empty"
    assert isinstance(samples, int),            "samples_trained must be int"
    assert samples == 5_000,                    f"Expected 5000 samples, got {samples}"
    assert isinstance(time_secs, float),        "time_seconds must be float"
    assert isinstance(loss, float),             "final_loss must be float"
    assert loss > 0,                            "Loss should be positive"

    # All tensors must be on CPU
    for k, v in state_dict.items():
        assert v.device.type == "cpu", f"Weight '{k}' is not on CPU — will break coordinator"

    # ── Test weight serialisation ─────────────────────────────────────────────
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        save_weights(state_dict, tmp.name)
        model2  = TransformerClassifier()
        model2  = load_weights(tmp.name, model2)
        state2  = model2.state_dict()
        for k in state_dict:
            assert torch.allclose(state_dict[k], state2[k]), f"Weight mismatch after save/load: {k}"
        os.unlink(tmp.name)

    print()
    print("=" * PRINT_WIDTH)
    print("  All checks passed.")
    print(f"  samples_trained : {samples:,}")
    print(f"  time_seconds    : {time_secs:.1f}s")
    print(f"  final_loss      : {loss:.4f}")
    print(f"  state_dict keys : {len(state_dict)}")
    print(f"  Weights on CPU  : ✓")
    print(f"  Save/load round-trip : ✓")
    print("=" * PRINT_WIDTH)