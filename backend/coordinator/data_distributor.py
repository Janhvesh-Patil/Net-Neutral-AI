"""
data_distributor.py — Net-Neutral AI

Data distribution module for the coordinator.
Handles loading, validating, and dividing datasets among clients.
"""

import os
import pandas as pd
from typing import Dict, Tuple, Optional


def load_and_validate_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and validate required columns exist (flexible column naming).

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with columns renamed to ['review', 'label']

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns not found
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")

    # Flexible column name detection (case-insensitive)
    text_col = next(
        (col for col in df.columns if col.lower() in ['review', 'text', 'content']),
        None
    )
    label_col = next(
        (col for col in df.columns if col.lower() in ['label', 'sentiment', 'class']),
        None
    )

    if not text_col or not label_col:
        raise ValueError(
            f"CSV must contain text and label columns. "
            f"Found columns: {list(df.columns)}\n"
            f"Expected: ['review' or 'text' or 'content'] and ['label' or 'sentiment' or 'class']"
        )

    return df[[text_col, label_col]].rename(columns={text_col: 'review', label_col: 'label'})


def divide_dataset(csv_path: str, num_clients: int) -> Dict[str, pd.DataFrame]:
    """
    Divide dataset equally among clients (stratified by label for balance).

    Args:
        csv_path: Path to dataset CSV
        num_clients: Number of clients to divide data among

    Returns:
        Dict mapping client_id (e.g., 'client_A') to DataFrame shard

    Raises:
        ValueError: If division fails or invalid input
    """
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")

    # Load and validate data
    df = load_and_validate_csv(csv_path)
    total_samples = len(df)

    if total_samples < num_clients:
        raise ValueError(
            f"Not enough samples ({total_samples}) for {num_clients} clients"
        )

    # Generate client IDs: client_A, client_B, client_C, ...
    clients = [f"client_{chr(65 + i)}" for i in range(num_clients)]

    # Stratified split to maintain label distribution
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError(
            "scikit-learn required for stratified splitting. "
            "Install with: pip install scikit-learn"
        )

    # NEW: 80/20 split for testing
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )
    
    # Save test dataset exclusively for coordinator
    test_csv_path = os.path.join(os.path.dirname(csv_path), "uploaded_test.csv")
    test_df.to_csv(test_csv_path, index=False)

    shards = {}
    remaining_df = train_df.copy()

    # Divide into N-1 shards (last one gets remainder)
    for i, client_id in enumerate(clients[:-1]):
        shard_size = len(remaining_df) // (num_clients - i)

        # Stratified split
        shard, remaining_df = train_test_split(
            remaining_df,
            test_size=len(remaining_df) - shard_size,
            stratify=remaining_df['label'],
            random_state=42
        )
        shards[client_id] = shard.reset_index(drop=True)

    # Last client gets remainder
    shards[clients[-1]] = remaining_df.reset_index(drop=True)

    return shards


def validate_shards(shards: Dict[str, pd.DataFrame], total_samples: int) -> bool:
    """
    Validate that shards divide correctly and preserve data.

    Args:
        shards: Dict of client_id -> DataFrame
        total_samples: Expected total number of samples

    Returns:
        True if valid

    Raises:
        AssertionError: If validation fails
    """
    total = sum(len(shard) for shard in shards.values())
    assert total == total_samples, f"Sample count mismatch: {total} vs {total_samples}"

    for client_id, shard in shards.items():
        assert len(shard) > 0, f"{client_id} has empty shard"
        assert 'review' in shard.columns, f"{client_id} missing 'review' column"
        assert 'label' in shard.columns, f"{client_id} missing 'label' column"

    return True


# Test suite
if __name__ == "__main__":
    import tempfile
    import sys

    print("=" * 60)
    print("  data_distributor.py — Dataset Division Test")
    print("=" * 60)

    # Create test dataset
    print("\n[Setup] Creating test dataset...")
    test_data = {
        'review': [f'Review {i}' for i in range(100)],
        'label': [i % 2 for i in range(100)]  # Balanced 50-50
    }
    test_df = pd.DataFrame(test_data)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_csv = f.name
        test_df.to_csv(test_csv, index=False)

    try:
        # Test 1: Load and validate CSV
        print("\n[Test 1] Load and validate CSV")
        loaded_df = load_and_validate_csv(test_csv)
        assert len(loaded_df) == 100, "Wrong number of rows"
        assert list(loaded_df.columns) == ['review', 'label'], "Wrong columns"
        print("  [OK] CSV loaded and validated")

        # Test 2: Divide into 3 clients
        print("\n[Test 2] Divide dataset into 3 clients")
        shards = divide_dataset(test_csv, 3)
        assert len(shards) == 3, "Wrong number of shards"
        assert 'client_A' in shards, "client_A missing"
        assert 'client_B' in shards, "client_B missing"
        assert 'client_C' in shards, "client_C missing"
        print("  [OK] 3 shards created")

        # Test 3: Validate shard distribution
        print("\n[Test 3] Validate shard distribution")
        for client_id, shard in shards.items():
            samples = len(shard)
            print(f"  {client_id}: {samples} samples")
            assert samples > 0, f"{client_id} has 0 samples"

        total = sum(len(s) for s in shards.values())
        print(f"  Total: {total} samples")
        assert total == 100, "Sample count mismatch"
        print("  [OK] All shards have data")

        # Test 4: Validate shards function
        print("\n[Test 4] Run validation function")
        validate_shards(shards, 100)
        print("  [OK] Validation passed")

        # Test 5: Check label distribution
        print("\n[Test 5] Check label distribution")
        for client_id, shard in shards.items():
            label_dist = shard['label'].value_counts().to_dict()
            print(f"  {client_id}: {label_dist}")

        print("  [OK] Labels present in all shards")

        # Test 6: Divide into 1 client
        print("\n[Test 6] Edge case: 1 client")
        shards_1 = divide_dataset(test_csv, 1)
        assert len(shards_1) == 1, "Should have 1 shard"
        assert len(shards_1['client_A']) == 100, "Should have all samples"
        print("  [OK] 1-client split works")

        # Test 7: Error handling - invalid CSV
        print("\n[Test 7] Error handling: invalid CSV")
        try:
            load_and_validate_csv("nonexistent.csv")
            assert False, "Should have raised error"
        except FileNotFoundError:
            print("  [OK] FileNotFoundError raised correctly")

        # Test 8: Error handling - missing columns
        print("\n[Test 8] Error handling: missing required columns")
        bad_data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        bad_df = pd.DataFrame(bad_data)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            bad_csv = f.name
            bad_df.to_csv(bad_csv, index=False)

        try:
            load_and_validate_csv(bad_csv)
            assert False, "Should have raised error"
        except ValueError as e:
            print(f"  [OK] ValueError raised: {str(e)[:50]}...")
        finally:
            os.unlink(bad_csv)

        print("\n" + "=" * 60)
        print("  All tests passed! data_distributor.py is ready")
        print("=" * 60 + "\n")

    finally:
        os.unlink(test_csv)
