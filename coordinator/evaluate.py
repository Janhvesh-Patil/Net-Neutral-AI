"""
evaluate.py — Net-Neutral AI
Model evaluation — accuracy, precision, recall, F1.

Design: pure functions, no printing, no hardcoded paths.
Callers (server.py, pretrain.py) handle all output formatting.

Spec reference: TRD Section 6.2
    - Evaluation metric : Accuracy (%) — primary
    - Additional metrics: Precision, Recall, F1 (per class + macro)
    - Validation set    : 2,000 samples held on coordinator
    - Called after      : every FedAvg round + pretrain baseline verification

    IMPORTANT RUN : python client/evaluate.py data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from dataclasses import dataclass, field


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """
    Holds all evaluation metrics for one evaluation run.
    Passed back to the caller — server.py or pretrain.py formats the output.
    """
    # Primary metric — what judges see
    accuracy: float                          # 0.0 to 1.0

    # Per-class metrics
    precision: Dict[int, float]              # {0: float, 1: float}
    recall:    Dict[int, float]              # {0: float, 1: float}
    f1:        Dict[int, float]              # {0: float, 1: float}

    # Macro averages (mean across both classes)
    macro_precision: float
    macro_recall:    float
    macro_f1:        float

    # Support (number of samples per class in eval set)
    support: Dict[int, int]                  # {0: int, 1: int}

    # Loss
    avg_loss: float

    # Counts for transparency
    total_samples: int
    correct:       int

    # Class names for readable output
    class_names: Dict[int, str] = field(
        default_factory=lambda: {0: "negative", 1: "positive"}
    )

    @property
    def accuracy_pct(self) -> float:
        """Accuracy as a percentage (0–100). Used in terminal output."""
        return self.accuracy * 100

    @property
    def accuracy_delta_str(self, previous: float = None) -> str:
        """Formatted accuracy delta string. Pass previous round accuracy."""
        return f"{self.accuracy_pct:.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
# CORE EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model:      nn.Module,
    dataloader: DataLoader,
    device:     torch.device = None,
) -> EvalResult:
    """
    Evaluate a model on a DataLoader. Returns full metrics.
    Does NOT print anything — caller handles output.

    Args:
        model      : TransformerClassifier (or any nn.Module with 2-class output)
        dataloader : validation DataLoader — typically 2,000 samples
        device     : torch.device to run on. If None, auto-detects.

    Returns:
        EvalResult dataclass with all metrics populated
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # ── Accumulators ──────────────────────────────────────────────────────────
    total_loss  = 0.0
    num_batches = 0
    all_preds   = []
    all_labels  = []

    # ── Inference loop ────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch_ids, batch_labels in dataloader:
            batch_ids    = batch_ids.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_ids)
            loss   = criterion(logits, batch_labels)

            total_loss  += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())

    # Move model back to CPU to free GPU memory
    model.cpu()

    # ── Compute metrics from raw predictions ──────────────────────────────────
    result = _compute_metrics(
        predictions = all_preds,
        labels      = all_labels,
        avg_loss    = total_loss / max(num_batches, 1),
    )
    return result


def _compute_metrics(
    predictions: list,
    labels:      list,
    avg_loss:    float,
) -> EvalResult:
    """
    Compute all classification metrics from raw prediction and label lists.
    Pure Python — no sklearn dependency, no extra imports.

    Metrics computed:
        - Accuracy  : (TP + TN) / total
        - Precision : TP / (TP + FP)  per class
        - Recall    : TP / (TP + FN)  per class
        - F1        : 2 * P * R / (P + R)  per class
        - Macro avg : mean of per-class metrics
    """
    classes = [0, 1]
    total   = len(labels)
    correct = sum(p == l for p, l in zip(predictions, labels))

    # ── Confusion matrix counts per class ─────────────────────────────────────
    # For class c:
    #   TP = predicted c,  actual c
    #   FP = predicted c,  actual not c
    #   FN = predicted not c,  actual c
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}

    for pred, label in zip(predictions, labels):
        for c in classes:
            if pred == c and label == c:
                tp[c] += 1
            elif pred == c and label != c:
                fp[c] += 1
            elif pred != c and label == c:
                fn[c] += 1

    # ── Per-class precision, recall, F1 ───────────────────────────────────────
    precision = {}
    recall    = {}
    f1        = {}
    support   = {}

    for c in classes:
        # Support = total actual samples of class c
        support[c] = sum(1 for l in labels if l == c)

        # Precision: avoid divide-by-zero if model never predicts class c
        precision[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0

        # Recall: avoid divide-by-zero if class c has no samples
        recall[c] = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0

        # F1: harmonic mean of precision and recall
        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        else:
            f1[c] = 0.0

    # ── Macro averages ────────────────────────────────────────────────────────
    macro_precision = sum(precision.values()) / len(classes)
    macro_recall    = sum(recall.values())    / len(classes)
    macro_f1        = sum(f1.values())        / len(classes)

    return EvalResult(
        accuracy         = correct / total,
        precision        = precision,
        recall           = recall,
        f1               = f1,
        macro_precision  = macro_precision,
        macro_recall     = macro_recall,
        macro_f1         = macro_f1,
        support          = support,
        avg_loss         = avg_loss,
        total_samples    = total,
        correct          = correct,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# Called by server.py and pretrain.py to format EvalResult for terminal output
# ─────────────────────────────────────────────────────────────────────────────

PRINT_WIDTH = 60

def _line(char="─"):
    return char * PRINT_WIDTH

def format_eval_result(
    result:        EvalResult,
    round_num:     int   = None,
    total_rounds:  int   = None,
    prev_accuracy: float = None,   # previous round accuracy (0.0–1.0) for delta
) -> str:
    """
    Format an EvalResult into a clean terminal string block.
    Returns a string — caller prints it.

    Args:
        result        : EvalResult from evaluate()
        round_num     : current round number (None for pretrain)
        total_rounds  : total rounds (None for pretrain)
        prev_accuracy : previous round's accuracy for delta display

    Returns:
        Formatted multi-line string ready for print()
    """
    lines = []
    W = PRINT_WIDTH

    def row(label, value, highlight=False):
        if highlight:
            # Highlight with surrounding markers for key metric
            lines.append(f"  {'★ ' + label:<28}{value}  ★")
        else:
            lines.append(f"  {label:<28}{value}")

    # ── Header ────────────────────────────────────────────────────────────────
    if round_num is not None:
        lines.append(_line("═"))
        lines.append(f"  COORDINATOR   |   Global Evaluation   |   Round {round_num} / {total_rounds}")
    else:
        lines.append(_line("═"))
        lines.append(f"  COORDINATOR   |   Baseline Evaluation")
    lines.append(_line("═"))

    # ── Primary metric — accuracy (highlighted) ───────────────────────────────
    lines.append("")
    acc_str = f"{result.accuracy_pct:.2f}%"
    if prev_accuracy is not None:
        delta     = result.accuracy - prev_accuracy
        delta_str = f"  ({'+' if delta >= 0 else ''}{delta*100:.2f}% vs last round)"
        acc_str  += delta_str
    row("Global Accuracy", acc_str, highlight=True)
    row("Correct / Total", f"{result.correct:,} / {result.total_samples:,}")
    row("Eval loss",       f"{result.avg_loss:.4f}")
    lines.append("")
    lines.append(_line("─"))

    # ── Per-class metrics ─────────────────────────────────────────────────────
    lines.append(f"  {'Class':<16}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    lines.append(_line("─"))
    for c in [0, 1]:
        name = result.class_names[c]
        lines.append(
            f"  {name:<16}"
            f"{result.precision[c]:>10.4f}"
            f"{result.recall[c]:>10.4f}"
            f"{result.f1[c]:>10.4f}"
            f"{result.support[c]:>10,}"
        )
    lines.append(_line("─"))
    lines.append(
        f"  {'macro avg':<16}"
        f"{result.macro_precision:>10.4f}"
        f"{result.macro_recall:>10.4f}"
        f"{result.macro_f1:>10.4f}"
        f"{result.total_samples:>10,}"
    )
    lines.append(_line("═"))
    lines.append("")

    return "\n".join(lines)


def format_accuracy_history(history: list) -> str:
    """
    Format a list of (round_num, accuracy_float) tuples into a summary table.
    Called by server.py at the end of all rounds.

    Args:
        history : list of (round_num, accuracy) e.g. [(1, 0.714), (2, 0.738)]

    Returns:
        Formatted string
    """
    lines = []
    lines.append(_line("═"))
    lines.append("  ACCURACY HISTORY")
    lines.append(_line("─"))
    lines.append(f"  {'Round':<10}{'Accuracy':>12}{'Delta':>12}")
    lines.append(_line("─"))

    for i, (rnd, acc) in enumerate(history):
        if i == 0:
            delta_str = "  (baseline)"
        else:
            delta = acc - history[i-1][1]
            delta_str = f"  {'+' if delta >= 0 else ''}{delta*100:.2f}%"
        lines.append(f"  {rnd:<10}{acc*100:>11.2f}%{delta_str}")

    lines.append(_line("═"))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "client"))

    from model import TransformerClassifier
    from data  import setup_data, get_validation_dataloader

    PRINT_WIDTH = 60
    print("=" * PRINT_WIDTH)
    print("  evaluate.py sanity check")
    print("=" * PRINT_WIDTH)

    data_dir   = sys.argv[1] if len(sys.argv) > 1 else None
    vocab_path = os.path.join(data_dir, "vocab.json") if data_dir else None

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[evaluate.py] Loading validation data...")
    _, _, test_texts, test_labels, vocab = setup_data(
        data_dir   = data_dir,
        vocab_path = vocab_path,
        save_vocab = False,
    )

    val_loader = get_validation_dataloader(test_texts, test_labels, vocab)

    # ── Test 1: random model (untrained) ──────────────────────────────────────
    print("[evaluate.py] Evaluating untrained model (expect ~50% accuracy)...")
    model  = TransformerClassifier()
    result = evaluate(model, val_loader)

    print("\n--- Untrained model ---")
    print(format_eval_result(result, round_num=None))

    assert 0.40 <= result.accuracy <= 0.65, \
        f"Untrained model accuracy {result.accuracy:.3f} out of expected range"
    assert result.total_samples == 2000, \
        f"Expected 2000 samples, got {result.total_samples}"
    assert 0 in result.precision and 1 in result.precision, \
        "Missing class keys in precision dict"

    # ── Test 2: accuracy history formatting ───────────────────────────────────
    fake_history = [(1, 0.673), (2, 0.714), (3, 0.738), (4, 0.759), (5, 0.791)]
    print(format_accuracy_history(fake_history))

    # ── Test 3: delta display ─────────────────────────────────────────────────
    result2 = evaluate(model, val_loader)
    print("--- With delta (same model, delta should be 0.00%) ---")
    print(format_eval_result(result2, round_num=2, total_rounds=5,
                              prev_accuracy=result.accuracy))

    print("=" * PRINT_WIDTH)
    print("  All checks passed. evaluate.py is ready.")
    print("=" * PRINT_WIDTH)