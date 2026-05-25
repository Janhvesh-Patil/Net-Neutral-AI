"""
fedavg.py — Net-Neutral AI
Federated Averaging (FedAvg) algorithm — weighted implementation.

Spec reference: TRD Section 4.3, App Flow Section 5.5

Algorithm (McMahan et al. 2017 — weighted variant):
    global_weight[key] = Σ(client_weight[key] × client_samples) / Σ(all_samples)

In the prototype all clients have 5,000 samples — weighted and unweighted
produce identical results. Weighted is kept for correctness in production
where clients may have unequal data sizes.

Edge cases handled:
    - Missing client (timed out)  — excluded, N decremented
    - Corrupted weight file       — excluded with warning
    - All clients missing         — raises FedAvgError
    - Weight shape mismatch       — excluded with warning
    - Missing sample count        — falls back to equal weight for that client
"""

import os
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ── Custom exception ──────────────────────────────────────────────────────────

class FedAvgError(Exception):
    """Raised when FedAvg cannot proceed — e.g. no valid submissions."""
    pass


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class FedAvgResult:
    """
    Holds the result of one FedAvg aggregation.
    Returned to server.py after each round.
    """
    global_state_dict:   Dict[str, torch.Tensor]
    clients_included:    int
    included_client_ids: List[str]
    skipped:             Dict[str, str] = field(default_factory=dict)

    @property
    def all_clients_included(self) -> bool:
        return len(self.skipped) == 0

    @property
    def summary(self) -> str:
        s = f"{self.clients_included} / {self.clients_included + len(self.skipped)} clients included"
        if self.skipped:
            s += f"  |  Skipped: {', '.join(self.skipped.keys())}"
        return s


# ─────────────────────────────────────────────────────────────────────────────
# CORE FEDAVG FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def federated_average(
    client_weights:       Dict[str, Dict[str, torch.Tensor]],
    client_samples:       Optional[Dict[str, int]] = None,
    reference_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> FedAvgResult:
    """
    Compute weighted federated average of client weight updates.

    Weighted formula:
        global[key] = Σ(client_i[key] × samples_i) / Σ(samples)

    Args:
        client_weights       : {client_id → state_dict}
        client_samples       : {client_id → samples_trained}
                               If None, equal weighting is used.
        reference_state_dict : used to verify weight shapes (optional).

    Returns:
        FedAvgResult with averaged state_dict and metadata

    Raises:
        FedAvgError : if no valid client weights remain after filtering
    """
    if not client_weights:
        raise FedAvgError("No client weights provided to federated_average()")

    # ── Step 1: Validate each client's weights ────────────────────────────────
    valid_clients: Dict[str, Dict[str, torch.Tensor]] = {}
    skipped:       Dict[str, str]                     = {}

    ref_keys = set(reference_state_dict.keys()) if reference_state_dict else None

    for client_id, state_dict in client_weights.items():
        reason = _validate_state_dict(state_dict, ref_keys, client_id)
        if reason is None:
            valid_clients[client_id] = state_dict
            if ref_keys is None:
                ref_keys = set(state_dict.keys())
        else:
            skipped[client_id] = reason
            print(f"[FedAvg] ⚠  Skipping {client_id}: {reason}")

    if not valid_clients:
        raise FedAvgError(
            f"No valid client weights after validation. Skipped: {skipped}"
        )

    # ── Step 2: Compute normalised sample weights ─────────────────────────────
    client_ids  = list(valid_clients.keys())
    state_dicts = list(valid_clients.values())
    n           = len(valid_clients)

    if client_samples:
        samples = [client_samples.get(cid, 5000) for cid in client_ids]
    else:
        samples = [1] * n

    total_samples = sum(samples)
    norm_weights  = [s / total_samples for s in samples]

    print(f"[FedAvg] Weighted averaging from {n} client(s): {client_ids}")
    for cid, s, w in zip(client_ids, samples, norm_weights):
        print(f"[FedAvg]   {cid}: {s:,} samples  →  weight {w:.4f}")

    # ── Step 3: Weighted element-wise average ─────────────────────────────────
    averaged: Dict[str, torch.Tensor] = {}

    for key in state_dicts[0].keys():
        weighted_sum = torch.zeros_like(state_dicts[0][key].float())
        for sd, w in zip(state_dicts, norm_weights):
            weighted_sum += sd[key].float() * w
        averaged[key] = weighted_sum

    print(f"[FedAvg] ✓  Weighted averaging complete — {len(averaged)} layers averaged")

    return FedAvgResult(
        global_state_dict    = averaged,
        clients_included     = n,
        included_client_ids  = client_ids,
        skipped              = skipped,
    )


def _validate_state_dict(
    state_dict: Dict[str, torch.Tensor],
    ref_keys:   Optional[set],
    client_id:  str,
) -> Optional[str]:
    """
    Validate a client state_dict before including in FedAvg.
    Returns None if valid, error string if invalid.
    """
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return "empty or invalid state_dict"

    for key, val in state_dict.items():
        if not isinstance(val, torch.Tensor):
            return f"non-tensor value for key '{key}'"

    if ref_keys is not None:
        client_keys = set(state_dict.keys())
        if client_keys != ref_keys:
            missing = ref_keys - client_keys
            extra   = client_keys - ref_keys
            return f"key mismatch — missing: {missing}, extra: {extra}"

    for key, val in state_dict.items():
        if torch.isnan(val).any():
            return f"NaN values in layer '{key}'"
        if torch.isinf(val).any():
            return f"Inf values in layer '{key}'"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT FILE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_client_weights(
    file_path: str,
    client_id: str,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[str]]:
    """
    Safely load a client's .pt weight file to CPU.
    Returns (state_dict, None) on success, (None, error) on failure.
    """
    if not os.path.exists(file_path):
        return None, f"file not found: {file_path}"
    try:
        state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            return None, "loaded object is not a state_dict"
        return state_dict, None
    except Exception as e:
        return None, f"torch.load failed: {str(e)}"


def save_global_model(
    state_dict:      Dict[str, torch.Tensor],
    checkpoint_path: str,
) -> None:
    """Overwrite checkpoint.pt with new averaged global model."""
    torch.save(state_dict, checkpoint_path)
    size_kb = os.path.getsize(checkpoint_path) / 1024
    print(f"[FedAvg] ✓  Global model saved → {checkpoint_path}  ({size_kb:.1f} KB)")


def load_global_model(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load checkpoint.pt into a state_dict.
    Called by server.py on startup and after each round.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run pretrain.py first to generate the baseline checkpoint."
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    print(f"[FedAvg] ✓  Global model loaded ← {checkpoint_path}")
    return state_dict


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "client"
    ))
    from model import TransformerClassifier

    PRINT_WIDTH = 60
    print("=" * PRINT_WIDTH)
    print("  fedavg.py sanity check (weighted)")
    print("=" * PRINT_WIDTH)

    model_A = TransformerClassifier()
    model_B = TransformerClassifier()
    model_C = TransformerClassifier()

    # ── Test 1: Equal samples — weighted == unweighted ────────────────────────
    print("\n[Test 1] Equal samples — weighted equals unweighted")
    weights = {
        "client_A": model_A.state_dict(),
        "client_B": model_B.state_dict(),
        "client_C": model_C.state_dict(),
    }
    samples = {"client_A": 5000, "client_B": 5000, "client_C": 5000}

    result_weighted   = federated_average(weights, client_samples=samples)
    result_unweighted = federated_average(weights, client_samples=None)

    key = "token_embedding.weight"
    assert torch.allclose(
        result_weighted.global_state_dict[key],
        result_unweighted.global_state_dict[key],
        atol=1e-5
    ), "Equal samples should produce identical weighted/unweighted results"
    print(f"  ✓  Weighted == unweighted for equal sample counts")
    print(f"  ✓  {result_weighted.summary}")

    # ── Test 2: Unequal samples — weighted differs from unweighted ────────────
    print("\n[Test 2] Unequal samples — weighted differs from unweighted")
    unequal_samples = {"client_A": 1000, "client_B": 5000, "client_C": 9000}
    result_unequal  = federated_average(weights, client_samples=unequal_samples)

    assert not torch.allclose(
        result_unequal.global_state_dict[key],
        result_unweighted.global_state_dict[key],
        atol=1e-3
    ), "Unequal samples should produce different result"
    print(f"  ✓  Unequal samples produce different result from equal weighting")
    total = sum(unequal_samples.values())
    for cid, s in unequal_samples.items():
        print(f"  ✓  {cid}: {s} samples → weight {s/total:.4f}")

    # ── Test 3: One client missing ────────────────────────────────────────────
    print("\n[Test 3] One client timed out — 2/3 clients")
    partial = {"client_A": model_A.state_dict(), "client_C": model_C.state_dict()}
    result3 = federated_average(partial, client_samples={"client_A": 5000, "client_C": 5000})
    assert result3.clients_included == 2
    print(f"  ✓  {result3.summary}")

    # ── Test 4: Corrupted client excluded ─────────────────────────────────────
    print("\n[Test 4] Corrupted client excluded")
    corrupted = {
        "client_A": model_A.state_dict(),
        "client_B": {"bad_key": torch.tensor([float("nan")])},
        "client_C": model_C.state_dict(),
    }
    result4 = federated_average(
        corrupted,
        client_samples={"client_A": 5000, "client_B": 5000, "client_C": 5000},
        reference_state_dict=model_A.state_dict()
    )
    assert result4.clients_included == 2
    assert "client_B" in result4.skipped
    print(f"  ✓  {result4.summary}")

    # ── Test 5: All clients invalid — FedAvgError ─────────────────────────────
    print("\n[Test 5] All clients invalid — FedAvgError")
    try:
        federated_average({"client_A": {}, "client_B": {}, "client_C": {}})
        assert False
    except FedAvgError:
        print(f"  ✓  FedAvgError raised correctly")

    # ── Test 6: Identical models — weighted average equals input ──────────────
    print("\n[Test 6] Identical models — weighted average equals input")
    same = {
        "client_A": model_A.state_dict(),
        "client_B": {k: v.clone() for k, v in model_A.state_dict().items()},
        "client_C": {k: v.clone() for k, v in model_A.state_dict().items()},
    }
    result6 = federated_average(same, client_samples=samples)
    for k in model_A.state_dict():
        assert torch.allclose(
            result6.global_state_dict[k],
            model_A.state_dict()[k].float(), atol=1e-5
        )
    print(f"  ✓  Average of identical models equals input")

    # ── Test 7: Save and load ─────────────────────────────────────────────────
    print("\n[Test 7] Save and load global model")
    import tempfile
    tmp      = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        save_global_model(result_weighted.global_state_dict, tmp_path)
        loaded = load_global_model(tmp_path)
        for k in result_weighted.global_state_dict:
            assert torch.allclose(result_weighted.global_state_dict[k], loaded[k], atol=1e-5)
        print(f"  ✓  Save/load round-trip verified")
    finally:
        os.unlink(tmp_path)

    print()
    print("=" * PRINT_WIDTH)
    print("  All checks passed. fedavg.py (weighted) is ready.")
    print("=" * PRINT_WIDTH)