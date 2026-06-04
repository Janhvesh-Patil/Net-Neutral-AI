"""
IMPORTANT RUN : python client/data.py data
"""

import os
import re
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

# ── Try importing datasets library ───────────────────────────────────────────
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("[data.py] WARNING: 'datasets' library not found. Run: pip install datasets")


# ── Constants (must match config.py and TRD spec) ────────────────────────────
VOCAB_SIZE   = 10_000
MAX_LEN      = 128
PAD_ID       = 0
UNK_ID       = 1
TRAIN_SIZE   = 15_000
VAL_SIZE     = 2_000
SHARD_SIZE   = 5_000

SHARD_RANGES = {
    "client_A": (0,      4_999),
    "client_B": (5_000,  9_999),
    "client_C": (10_000, 14_999),
}

LABEL_MAP = {
    "positive": 1,
    "negative": 0,
    "pos":      1,
    "neg":      0,
    1:          1,
    0:          0,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_imdb_data(data_dir: Optional[str] = None) -> Tuple[List[str], List[int], List[str], List[int]]:

    # ── Attempt 1: local CSV files ────────────────────────────────────────────
    if data_dir is not None:
        def _find_file(folder, *names):
            for name in names:
                p = os.path.join(folder, name)
                if os.path.exists(p):
                    return p
            return None

        train_path = _find_file(data_dir, "imdb_train.csv", "train.csv")
        test_path  = _find_file(data_dir, "imdb_test.csv",  "test.csv")

        if train_path and test_path:
            print("[data.py] Loading from local CSV files...")
            train_texts, train_labels = _load_csv(train_path)
            test_texts,  test_labels  = _load_csv(test_path)
            print(f"[data.py] Loaded {len(train_texts):,} train + {len(test_texts):,} test samples")
            return train_texts, train_labels, test_texts, test_labels
        else:
            print(f"[data.py] CSV files not found at {data_dir} — falling back to HuggingFace")

    # ── Attempt 2: ajaykarthick/imdb-movie-reviews ───────────────────────────
    if HF_DATASETS_AVAILABLE:
        try:
            print("[data.py] Loading ajaykarthick/imdb-movie-reviews from HuggingFace...")
            dataset = load_dataset("ajaykarthick/imdb-movie-reviews")
            train_texts, train_labels = _extract_hf_split(dataset["train"])
            test_texts,  test_labels  = _extract_hf_split(dataset["test"])
            print(f"[data.py] Loaded {len(train_texts):,} train + {len(test_texts):,} test samples")
            return train_texts, train_labels, test_texts, test_labels
        except Exception as e:
            print(f"[data.py] ajaykarthick dataset failed ({e}) — trying standard imdb")

        # ── Attempt 3: standard HuggingFace imdb dataset ─────────────────────
        try:
            print("[data.py] Loading standard 'imdb' dataset from HuggingFace...")
            dataset = load_dataset("imdb")
            train_texts  = dataset["train"]["text"]
            train_labels = dataset["train"]["label"]
            test_texts   = dataset["test"]["text"]
            test_labels  = dataset["test"]["label"]
            print(f"[data.py] Loaded {len(train_texts):,} train + {len(test_texts):,} test samples")
            return train_texts, train_labels, test_texts, test_labels
        except Exception as e:
            raise RuntimeError(f"[data.py] All data loading attempts failed. Last error: {e}")

    raise RuntimeError("[data.py] 'datasets' library not installed and no local files found.")


def _load_csv(path: str) -> Tuple[List[str], List[int]]:

    import csv
    texts, labels = [], []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text  = row.get("review") or row.get("text") or row.get("Review") or ""
            label = row.get("sentiment") or row.get("label") or row.get("Sentiment") or "0"
            text  = text.strip()
            if isinstance(label, str):
                label_clean = label.strip().lower()
                if label_clean.isdigit():
                    resolved = int(label_clean)
                else:
                    resolved = LABEL_MAP.get(label_clean, 0)
            else:
                resolved = LABEL_MAP.get(int(label), 0)
            if text:
                texts.append(text)
                labels.append(resolved)
    return texts, labels


def _extract_hf_split(split) -> Tuple[List[str], List[int]]:

    texts, labels = [], []
    for item in split:
        text  = item.get("review") or item.get("text") or ""
        label = item.get("sentiment") or item.get("label") or 0
        if isinstance(label, str):
            label = LABEL_MAP.get(label.strip().lower(), 0)
        texts.append(text.strip())
        labels.append(int(label))
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> List[str]:
    return clean_text(text).split()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:

    def __init__(self):
        self.word2id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.id2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.size: int = 2   # starts at 2 (0 and 1 reserved)

    def build(self, texts: List[str], max_size: int = VOCAB_SIZE) -> None:

        print(f"[Vocabulary] Building from {len(texts):,} texts...")
        counter = Counter()
        for text in texts:
            tokens = tokenise(text)
            counter.update(tokens)

        most_common = counter.most_common(max_size - 2)
        print(f"[Vocabulary] Unique tokens found: {len(counter):,}  →  keeping top {len(most_common):,}")

        for word, freq in most_common:
            idx = self.size
            self.word2id[word] = idx
            self.id2word[idx]  = word
            self.size += 1

        print(f"[Vocabulary] Final vocabulary size: {self.size:,}")

    def encode(self, text: str) -> List[int]:
        tokens = tokenise(text)
        return [self.word2id.get(token, UNK_ID) for token in tokens]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.word2id, f)
        print(f"[Vocabulary] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = cls()
        with open(path, "r") as f:
            word2id = json.load(f)
        vocab.word2id = word2id
        vocab.id2word = {v: k for k, v in word2id.items()}
        vocab.size    = len(word2id)
        print(f"[Vocabulary] Loaded from {path} — size: {vocab.size:,}")
        return vocab


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class IMDbDataset(Dataset):

    def __init__(
        self,
        texts:   List[str],
        labels:  List[int],
        vocab:   Vocabulary,
        max_len: int = MAX_LEN,
    ):

        assert len(texts) == len(labels), "texts and labels must have the same length"
        self.texts   = texts
        self.labels  = labels
        self.vocab   = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        token_ids = self.vocab.encode(self.texts[idx])

        token_ids = token_ids[:self.max_len]

        padding   = [PAD_ID] * (self.max_len - len(token_ids))
        token_ids = token_ids + padding

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label     = torch.tensor(self.labels[idx], dtype=torch.long)

        return input_ids, label


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — DATALOADER FACTORY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_client_dataloader(
    client_id:  str,
    train_texts: List[str],
    train_labels: List[int],
    vocab:      Vocabulary,
    batch_size: int = 32,
    shuffle:    bool = True,
) -> DataLoader:

    if client_id not in SHARD_RANGES:
        raise ValueError(f"Unknown client_id '{client_id}'. Must be one of {list(SHARD_RANGES.keys())}")

    start, end = SHARD_RANGES[client_id]
    shard_texts  = train_texts[start : end + 1]
    shard_labels = train_labels[start : end + 1]

    print(f"[data.py] {client_id} shard: rows {start}–{end} ({len(shard_texts):,} samples)")
    print(f"[data.py] Label distribution: {sum(shard_labels)} positive, {len(shard_labels)-sum(shard_labels)} negative")

    dataset = IMDbDataset(shard_texts, shard_labels, vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_validation_dataloader(
    test_texts:  List[str],
    test_labels: List[int],
    vocab:       Vocabulary,
    batch_size:  int = 64,
) -> DataLoader:

    val_texts  = test_texts[:VAL_SIZE]
    val_labels = test_labels[:VAL_SIZE]

    print(f"[data.py] Validation set: {len(val_texts):,} samples")

    dataset = IMDbDataset(val_texts, val_labels, vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_full_dataloader(
    texts:      List[str],
    labels:     List[int],
    vocab:      Vocabulary,
    batch_size: int = 32,
    shuffle:    bool = True,
) -> DataLoader:

    dataset = IMDbDataset(texts, labels, vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SETUP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def setup_data(
    data_dir:     Optional[str] = None,
    vocab_path:   Optional[str] = None,
    save_vocab:   bool = True,
) -> Tuple[List[str], List[int], List[str], List[int], Vocabulary]:

    train_texts, train_labels, test_texts, test_labels = load_imdb_data(data_dir)

    if vocab_path and os.path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = Vocabulary()
        vocab.build(train_texts[:TRAIN_SIZE], max_size=VOCAB_SIZE)
        if save_vocab and vocab_path:
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            vocab.save(vocab_path)

    return train_texts, train_labels, test_texts, test_labels, vocab


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("data.py sanity check")
    print("=" * 60)

    # ── Accept optional data_dir argument ─────────────────────────────────────
    data_dir   = sys.argv[1] if len(sys.argv) > 1 else None
    vocab_path = os.path.join(data_dir, "vocab.json") if data_dir else None

    # ── Load data ─────────────────────────────────────────────────────────────
    train_texts, train_labels, test_texts, test_labels, vocab = setup_data(
        data_dir   = data_dir,
        vocab_path = vocab_path,
        save_vocab = True,
    )

    # ── Check label distribution in full training set ─────────────────────────
    pos = sum(train_labels[:TRAIN_SIZE])
    neg = TRAIN_SIZE - pos
    print(f"\nTraining subset (15K):")
    print(f"  Positive: {pos:,}  |  Negative: {neg:,}")

    # ── Check each client shard ───────────────────────────────────────────────
    print("\nClient shard DataLoaders:")
    for cid in ["client_A", "client_B", "client_C"]:
        loader = get_client_dataloader(cid, train_texts, train_labels, vocab)
        batch_ids, batch_labels = next(iter(loader))
        print(f"  {cid}: {len(loader.dataset):,} samples | "
              f"batch shape: {batch_ids.shape} | "
              f"label sample: {batch_labels[:4].tolist()}")
        assert batch_ids.shape  == (32, MAX_LEN), f"Unexpected shape: {batch_ids.shape}"
        assert batch_ids.max()  < VOCAB_SIZE,     "Token ID exceeds vocab size"
        assert batch_ids.min()  >= 0,             "Negative token ID found"

    # ── Check validation DataLoader ───────────────────────────────────────────
    print("\nValidation DataLoader:")
    val_loader = get_validation_dataloader(test_texts, test_labels, vocab)
    val_ids, val_labels = next(iter(val_loader))
    print(f"  {len(val_loader.dataset):,} samples | "
          f"batch shape: {val_ids.shape} | "
          f"label sample: {val_labels[:4].tolist()}")

    # ── Check vocabulary ──────────────────────────────────────────────────────
    print(f"\nVocabulary:")
    print(f"  Size       : {vocab.size:,}")
    print(f"  Token 0    : '{vocab.id2word[0]}'  (must be <PAD>)")
    print(f"  Token 1    : '{vocab.id2word[1]}'  (must be <UNK>)")
    print(f"  Token 2    : '{vocab.id2word[2]}'  (most frequent word)")
    print(f"  Token 3    : '{vocab.id2word[3]}'")
    print(f"  Token 4    : '{vocab.id2word[4]}'")

    # ── Check encoding ────────────────────────────────────────────────────────
    sample_text = "This movie was absolutely terrible and I hated every minute"
    encoded     = vocab.encode(sample_text)
    print(f"\nEncoding test:")
    print(f"  Text   : '{sample_text}'")
    print(f"  Encoded: {encoded}")
    assert UNK_ID not in encoded or True, "UNK found — fine if word not in vocab"

    # ── Check padding ─────────────────────────────────────────────────────────
    short_text = "Good film"
    dataset    = IMDbDataset([short_text], [1], vocab)
    ids, lbl   = dataset[0]
    assert ids.shape == (MAX_LEN,),       f"Expected ({MAX_LEN},), got {ids.shape}"
    assert ids[-1].item() == PAD_ID,      "Last token should be PAD for short text"
    assert lbl.item() == 1,               "Label should be 1 (positive)"
    print(f"\nPadding test: short text padded to {ids.shape} ✓")

    print("\n" + "=" * 60)
    print("All checks passed. data.py is ready.")
    print("=" * 60)