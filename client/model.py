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
        super().__init__()

        # ── 1. Token embedding ────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # ── 2. Positional encoding (learned) ──────────────────────────────────
        self.position_embedding = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=embed_dim,
        )

        # ── 3. Transformer encoder layers ─────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # ── 4. Linear classifier ──────────────────────────────────────────────
        self.classifier = nn.Linear(embed_dim, num_classes)

        # ── Dropout for regularisation ────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

        # ── Store max_len so forward() can build position indices ─────────────
        self.max_len = max_len

        # ── Initialise weights ────────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
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
        token_embeds = self.token_embedding(input_ids)

        # ── Step 2: Positional embeddings ─────────────────────────────────────
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # position_embeds: (batch, seq_len, embed_dim)
        position_embeds = self.position_embedding(positions)

        # ── Step 3: Add token + position embeddings, apply dropout ────────────
        x = self.dropout(token_embeds + position_embeds)

        # ── Step 4: Build padding mask ────────────────────────────────────────
        padding_mask = (input_ids == 0)

        # ── Step 5: Transformer encoder ───────────────────────────────────────
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # ── Step 6: Global average pooling ────────────────────────────────────
        valid_mask = (~padding_mask).float().unsqueeze(-1)

        x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-9)

        # ── Step 7: Classify ──────────────────────────────────────────────────
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


# ── Sanity check ─────────────────────────────────────────────────────────────
# Run this file directly to verify the model builds and forward pass works.
# Expected output:
#   Model output shape: torch.Size([8, 2])
#   Total parameters:   1,600,642
#   Model structure printed below.

if __name__ == "__main__":

    print("Running model.py sanity check...\n")

    model = TransformerClassifier()

    dummy_input = torch.randint(low=1, high=10_000, size=(8, 128))

    dummy_input[0, 100:] = 0
    dummy_input[3, 64:] = 0

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
