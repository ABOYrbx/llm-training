import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=384,
        num_heads=6,
        num_layers=8,
        max_seq_len=512
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.token_emb(x) + self.pos_emb[:, :seq_len]
        x = self.transformer(x)
        return self.lm_head(x)
