"""
model.py — Net-Neutral AI
TransformerClassifier: 2-layer Transformer for binary sentiment classification

Spec reference: TRD Section 6.1
Architecture:
    1. Embedding layer         (vocab_size=10000, embed_dim=128)
    2. Positional encoding     (learned, max_len=128)
    3. TransformerEncoder x2   (nhead=4, dim_feedforward=256, dropout=0.1)
    4. Global average pooling  (mean across sequence dimension)
    5. Linear classifier       (128 → 2)

Input:  token IDs  — shape (batch_size, seq_len)
Output: raw logits — shape (batch_size, 2)
        class 0 = negative sentiment
        class 1 = positive sentiment
"""

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):

    def __init__(
            self,
            vocab_size: int = 10_000,
            embed_dim: int = 128,
            num_heads: int = 4,
            ffn_dim: int = 256,
            num_layers: int = 2,
            max_len: int = 128,
            dropout: float = 0.1,
            num_classes: int = 2,
    ):
        """
        Args:
            vocab_size  : number of unique tokens in vocabulary
            embed_dim   : dimension of token embeddings (d_model)
            num_heads   : number of attention heads in each encoder layer
            ffn_dim     : hidden dimension of the feedforward sublayer
            num_layers  : number of stacked TransformerEncoder layers
            max_len     : maximum sequence length (for positional encoding)
            dropout     : dropout probability applied inside encoder layers
            num_classes : output classes (2 for binary sentiment)
        """
        super().__init__()

        # ── 1. Token embedding ────────────────────────────────────────────────
        # Maps each integer token ID → a learned vector of size embed_dim.
        # padding_idx=0 means the padding token contributes zero to gradients.
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # ── 2. Positional encoding (learned) ──────────────────────────────────
        # Each position 0..max_len-1 gets its own learned vector of size embed_dim.
        # These are ADDED to the token embeddings so the model knows word order.
        # We store positions as a parameter so they are saved in state_dict.
        self.position_embedding = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=embed_dim,
        )

        # ── 3. Transformer encoder layers ─────────────────────────────────────
        # One TransformerEncoderLayer = multi-head self-attention + feedforward.
        # We stack num_layers of these using TransformerEncoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  # input/output size of each layer
            nhead=num_heads,  # must divide embed_dim evenly: 128/4 = 32 ✓
            dim_feedforward=ffn_dim,  # hidden size inside feedforward sublayer
            dropout=dropout,
            batch_first=True,  # input shape: (batch, seq, embed) not (seq, batch, embed)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # ── 4. Global average pooling ─────────────────────────────────────────
        # No parameters here — this is just an operation in forward().
        # We take the mean across the sequence dimension → (batch, embed_dim).

        # ── 5. Linear classifier ──────────────────────────────────────────────
        # Maps the pooled vector to logits for each class.
        self.classifier = nn.Linear(embed_dim, num_classes)

        # ── Dropout for regularisation ────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

        # ── Store max_len so forward() can build position indices ─────────────
        self.max_len = max_len

        # ── Initialise weights ────────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Xavier uniform initialisation for the classifier layer.
        Embedding weights use PyTorch's default (normal distribution).
        This gives the model a stable starting point for training.
        """
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids : LongTensor of shape (batch_size, seq_len)
                        Each value is a token ID in range [0, vocab_size).
                        Padded positions should be 0 (matches padding_idx).

        Returns:
            logits    : FloatTensor of shape (batch_size, 2)
                        Raw scores — pass through softmax for probabilities,
                        or use directly with CrossEntropyLoss during training.
        """
        batch_size, seq_len = input_ids.shape

        # Safety check: truncate silently if sequence exceeds max_len
        if seq_len > self.max_len:
            input_ids = input_ids[:, :self.max_len]
            seq_len = self.max_len

        # ── Step 1: Token embeddings ──────────────────────────────────────────
        # input_ids: (batch, seq_len)  →  token_embeds: (batch, seq_len, embed_dim)
        token_embeds = self.token_embedding(input_ids)

        # ── Step 2: Positional embeddings ─────────────────────────────────────
        # Build position indices [0, 1, 2, ..., seq_len-1] for every item in batch.
        # positions: (seq_len,)  →  expand to (batch_size, seq_len)
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # position_embeds: (batch, seq_len, embed_dim)
        position_embeds = self.position_embedding(positions)

        # ── Step 3: Add token + position embeddings, apply dropout ────────────
        # x: (batch, seq_len, embed_dim)
        x = self.dropout(token_embeds + position_embeds)

        # ── Step 4: Build padding mask ────────────────────────────────────────
        # Tells the attention mechanism to ignore padding tokens (id = 0).
        # src_key_padding_mask: True where token is padding → attention ignores it.
        # Shape: (batch, seq_len) — True means "ignore this position"
        padding_mask = (input_ids == 0)

        # ── Step 5: Transformer encoder ───────────────────────────────────────
        # x: (batch, seq_len, embed_dim)  →  x: (batch, seq_len, embed_dim)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # ── Step 6: Global average pooling ────────────────────────────────────
        # We want one vector per sample, not one per token.
        # Exclude padding positions from the mean to avoid diluting the signal.
        #
        # padding_mask is True for padding → we want the opposite for weighting.
        # valid_mask: (batch, seq_len, 1) — 1.0 for real tokens, 0.0 for padding
        valid_mask = (~padding_mask).float().unsqueeze(-1)

        # Zero out padding positions, sum, divide by number of real tokens
        x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-9)
        # x: (batch, embed_dim)

        # ── Step 7: Classify ──────────────────────────────────────────────────
        x = self.dropout(x)
        logits = self.classifier(x)
        # logits: (batch, 2)

        return logits


# ── Sanity check ─────────────────────────────────────────────────────────────
# Run this file directly to verify the model builds and forward pass works.
# Expected output:
#   Model output shape: torch.Size([8, 2])
#   Total parameters:   1,600,642
#   Model structure printed below.

if __name__ == "__main__":

    print("Running model.py sanity check...\n")

    # Build model with default hyperparameters from TRD
    model = TransformerClassifier()

    # Dummy input: batch of 8 sequences, each 128 tokens long
    dummy_input = torch.randint(low=1, high=10_000, size=(8, 128))

    # Include some padding tokens to test padding mask
    dummy_input[0, 100:] = 0  # last 28 tokens of first sample are padding
    dummy_input[3, 64:] = 0  # last 64 tokens of fourth sample are padding

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Output sample: {output[0]}")
    assert output.shape == (8, 2), f"Expected (8, 2), got {output.shape}"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # State dict check — this is what gets sent over the network in FedAvg
    state_dict = model.state_dict()
    print(f"\nstate_dict keys ({len(state_dict)} layers):")
    for k, v in state_dict.items():
        print(f"  {k:55s}  {str(v.shape):30s}  {v.dtype}")

    print("\nAll checks passed. model.py is ready.")
#%% raw
