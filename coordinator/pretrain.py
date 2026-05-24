"""
pretrain.py — Net-Neutral AI
One-time baseline training. Run ONCE before the demo on a GPU machine.

⚠️  DO NOT run on CPU — 15K samples x 5 epochs ≈ 40–50 minutes on CPU.
    Run on ENTC1's GPU machine — takes 5–8 minutes.

Purpose:
    Train TransformerClassifier on the full 15,000 sample dataset
    for 5 epochs. Save the result as coordinator/checkpoint.pt.
    This checkpoint is the starting point for Round 1 of federated training.

Why this matters:
    A randomly initialised model starts at ~50% accuracy (coin flip).
    A pre-trained baseline starts at ~65–70%.
    Federated rounds then push it toward 75–80%.
    Without this, the accuracy improvement curve is less convincing on camera.

Output:
    coordinator/checkpoint.pt  — saved model weights
    coordinator/pretrain_log.json — training history for reference

Usage:
    python coordinator/pretrain.py data
    python coordinator/pretrain.py data --epochs 5 --lr 0.001
"""

import sys
import os
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Path fix — allows imports from client/ ────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "client"))

from model import TransformerClassifier
from data import setup_data, get_full_dataloader
from evaluate import evaluate, format_eval_result, format_accuracy_history, EvalResult

# ── Output paths ──────────────────────────────────────────────────────────────
COORDINATOR_DIR  = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH  = os.path.join(COORDINATOR_DIR, "checkpoint.pt")
PRETRAIN_LOG     = os.path.join(COORDINATOR_DIR, "pretrain_log.json")

# ── Default hyperparameters ───────────────────────────────────────────────────
DEFAULT_EPOCHS = 5
DEFAULT_LR     = 1e-3
BATCH_SIZE     = 32
TRAIN_SIZE     = 15_000
PRINT_WIDTH    = 60


# ── Terminal helpers ──────────────────────────────────────────────────────────
def _line(char="─"): return char * PRINT_WIDTH
def _row(label, value): print(f"  {label:<28}{value}")
def _blank(): print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PRETRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(
    data_dir:   str,
    epochs:     int   = DEFAULT_EPOCHS,
    lr:         float = DEFAULT_LR,
    batch_size: int   = BATCH_SIZE,
) -> None:
    """
    Train TransformerClassifier on full 15K dataset and save checkpoint.

    Args:
        data_dir   : path to folder containing CSV files and vocab.json
        epochs     : number of training epochs
        lr         : learning rate for Adam optimiser
        batch_size : samples per batch
    """

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(_line("═"))
    print("  Net-Neutral AI   |   PRETRAIN")
    print("  Baseline checkpoint generation")
    print(_line("═"))
    _blank()

    if device.type == "cpu":
        print("  ⚠  WARNING: Running on CPU.")
        print("     Estimated time: 40–50 minutes.")
        print("     Strongly recommended: run on ENTC1's GPU (5–8 min).")
        print()
        response = input("  Continue on CPU anyway? (y/n): ").strip().lower()
        if response != "y":
            print("  Aborted. Transfer this script to a GPU machine and rerun.")
            sys.exit(0)
        _blank()
    else:
        print(f"  ✓  GPU detected: {torch.cuda.get_device_name(0)}")
        _blank()

    _row("Device",          str(device).upper())
    _row("Epochs",          str(epochs))
    _row("Learning rate",   str(lr))
    _row("Batch size",      str(batch_size))
    _row("Training samples",f"{TRAIN_SIZE:,}")
    _row("Checkpoint path", CHECKPOINT_PATH)
    _blank()

    # ── Load data ─────────────────────────────────────────────────────────────
    vocab_path = os.path.join(data_dir, "vocab.json")
    print(_line("─"))
    print("  Loading data...")
    print(_line("─"))

    train_texts, train_labels, test_texts, test_labels, vocab = setup_data(
        data_dir   = data_dir,
        vocab_path = vocab_path,
        save_vocab = False,
    )

    # Full 15K training set
    full_train_texts  = train_texts[:TRAIN_SIZE]
    full_train_labels = train_labels[:TRAIN_SIZE]

    train_loader = get_full_dataloader(
        texts      = full_train_texts,
        labels     = full_train_labels,
        vocab      = vocab,
        batch_size = batch_size,
        shuffle    = True,
    )

    # Validation loader for post-epoch evaluation
    from data import get_validation_dataloader
    val_loader = get_validation_dataloader(test_texts, test_labels, vocab)

    _row("Train batches",   f"{len(train_loader):,}")
    _row("Val samples",     f"{len(val_loader.dataset):,}")
    _blank()

    # ── Model ─────────────────────────────────────────────────────────────────
    print(_line("─"))
    print("  Initialising model...")
    print(_line("─"))
    model     = TransformerClassifier()
    model     = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    _row("Total parameters", f"{total_params:,}")
    _blank()

    # ── Training setup ────────────────────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler — reduces LR by 50% if val loss stops improving
    # Helps squeeze out extra accuracy in later epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=1, verbose=False
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(_line("═"))
    print("  Training")
    print(_line("═"))

    history       = []   # list of (epoch, train_loss, train_acc, val_accuracy)
    best_accuracy = 0.0
    total_start   = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start  = time.time()
        epoch_loss   = 0.0
        correct      = 0
        total        = 0
        num_batches  = 0

        model.train()
        for batch_ids, batch_labels in train_loader:
            batch_ids    = batch_ids.to(device)
            batch_labels = batch_labels.to(device)

            optimiser.zero_grad()
            logits = model(batch_ids)
            loss   = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss  += loss.item()
            preds        = logits.argmax(dim=1)
            correct     += (preds == batch_labels).sum().item()
            total       += batch_labels.size(0)
            num_batches += 1

        # ── Epoch metrics ─────────────────────────────────────────────────────
        train_loss = epoch_loss / num_batches
        train_acc  = correct / total * 100
        epoch_time = time.time() - epoch_start

        # ── Validation after every epoch ──────────────────────────────────────
        result     = evaluate(model.cpu(), val_loader)
        model      = model.to(device)  # move back after evaluate() puts it on CPU
        val_acc    = result.accuracy_pct

        # Update LR scheduler based on validation accuracy
        scheduler.step(result.accuracy)

        # Save best checkpoint
        if result.accuracy > best_accuracy:
            best_accuracy = result.accuracy
            torch.save(model.cpu().state_dict(), CHECKPOINT_PATH)
            model = model.to(device)
            saved_str = "  ✓ checkpoint saved"
        else:
            saved_str = ""

        # ── Print epoch summary ───────────────────────────────────────────────
        print(_line("─"))
        print(f"  Epoch {epoch} / {epochs}")
        print(_line("─"))
        _row("Train loss",      f"{train_loss:.4f}")
        _row("Train accuracy",  f"{train_acc:.2f}%")
        _row("Val accuracy",    f"{val_acc:.2f}%  {saved_str}")
        _row("Epoch time",      f"{epoch_time:.1f}s")
        current_lr = optimiser.param_groups[0]["lr"]
        _row("Learning rate",   f"{current_lr:.6f}")

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  2),
            "val_acc":    round(val_acc,    2),
            "epoch_time": round(epoch_time, 1),
            "lr":         current_lr,
        })

    # ── Training complete ─────────────────────────────────────────────────────
    total_time = time.time() - total_start
    _blank()
    print(_line("═"))
    print("  Pretraining Complete")
    print(_line("═"))
    _row("Total time",       f"{total_time:.1f}s  ({total_time/60:.1f} min)")
    _row("Best val accuracy",f"{best_accuracy*100:.2f}%")
    _row("Checkpoint saved", CHECKPOINT_PATH)
    _blank()

    # ── Final evaluation on best checkpoint ───────────────────────────────────
    print(_line("─"))
    print("  Final evaluation on best checkpoint")
    print(_line("─"))
    best_model = TransformerClassifier()
    best_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    final_result = evaluate(best_model, val_loader)
    print(format_eval_result(final_result, round_num=None))

    # ── Save training log ─────────────────────────────────────────────────────
    log = {
        "epochs":          epochs,
        "lr":              lr,
        "batch_size":      batch_size,
        "train_samples":   TRAIN_SIZE,
        "best_val_acc":    round(best_accuracy * 100, 2),
        "total_time_secs": round(total_time, 1),
        "device":          str(device),
        "history":         history,
        "final_metrics": {
            "accuracy":         round(final_result.accuracy_pct, 2),
            "macro_precision":  round(final_result.macro_precision, 4),
            "macro_recall":     round(final_result.macro_recall,    4),
            "macro_f1":         round(final_result.macro_f1,        4),
        }
    }
    with open(PRETRAIN_LOG, "w") as f:
        json.dump(log, f, indent=2)
    _row("Training log saved", PRETRAIN_LOG)
    _blank()

    # ── Accuracy history across epochs ────────────────────────────────────────
    epoch_history = [(h["epoch"], h["val_acc"] / 100) for h in history]
    print(format_accuracy_history(epoch_history))

    print(_line("═"))
    print("  checkpoint.pt is ready.")
    print("  Commit it to the repo before the demo.")
    print("  All clients will download it via GET /model on round 1.")
    print(_line("═"))
    _blank()


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT VERIFICATION UTILITY
# Run this to verify an existing checkpoint without retraining
# Usage: python coordinator/pretrain.py data --verify-only
# ─────────────────────────────────────────────────────────────────────────────

def verify_checkpoint(data_dir: str) -> None:
    """
    Verify an existing checkpoint.pt without retraining.
    Prints accuracy and full metrics. Fast — inference only.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[pretrain.py] No checkpoint found at {CHECKPOINT_PATH}")
        print("  Run pretrain.py without --verify-only to generate one.")
        sys.exit(1)

    print()
    print(_line("═"))
    print("  Net-Neutral AI   |   Checkpoint Verification")
    print(_line("═"))
    _blank()
    _row("Checkpoint", CHECKPOINT_PATH)
    _row("File size",  f"{os.path.getsize(CHECKPOINT_PATH) / 1024:.1f} KB")
    _blank()

    # Load vocab and validation data
    vocab_path = os.path.join(data_dir, "vocab.json")
    _, _, test_texts, test_labels, vocab = setup_data(
        data_dir   = data_dir,
        vocab_path = vocab_path,
        save_vocab = False,
    )
    from data import get_validation_dataloader
    val_loader = get_validation_dataloader(test_texts, test_labels, vocab)

    # Load checkpoint
    model = TransformerClassifier()
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        print("  ✓  Checkpoint loaded successfully")
    except Exception as e:
        print(f"  ✗  Checkpoint load failed: {e}")
        sys.exit(1)

    # Evaluate
    result = evaluate(model, val_loader)
    print(format_eval_result(result, round_num=None))

    # Check pretrain log if it exists
    if os.path.exists(PRETRAIN_LOG):
        with open(PRETRAIN_LOG) as f:
            log = json.load(f)
        print(_line("─"))
        print("  Training log summary:")
        _row("Trained for",    f"{log['epochs']} epochs")
        _row("Best val acc",   f"{log['best_val_acc']}%")
        _row("Total time",     f"{log['total_time_secs']}s")
        _row("Device used",    log['device'])
        print(_line("═"))

    if result.accuracy < 0.55:
        print("  ⚠  WARNING: Accuracy below 55%.")
        print("     Consider rerunning pretrain.py with more epochs.")
    else:
        print("  ✓  Checkpoint is demo-ready.")
    _blank()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Net-Neutral AI — pretrain baseline model")
    parser.add_argument("data_dir",      type=str,   help="Path to data folder (contains CSV + vocab.json)")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS, help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR,     help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE,     help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--verify-only", action="store_true",                help="Verify existing checkpoint without retraining")
    args = parser.parse_args()

    if args.verify_only:
        verify_checkpoint(args.data_dir)
    else:
        pretrain(
            data_dir   = args.data_dir,
            epochs     = args.epochs,
            lr         = args.lr,
            batch_size = args.batch_size,
        )