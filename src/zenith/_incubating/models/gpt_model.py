import torch
import torch.nn as nn
from .transformer_model import MultiHeadAttention, FeedForwardNetwork

class GPTBlock(nn.Module):
    """A single block for a GPT (decoder-only) model."""
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout):
        super(GPTBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForwardNetwork(embed_size, ff_hidden_dim, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # The first sub-layer is masked multi-head self-attention
        attention = self.attention(x, x, x, mask)
        x = self.dropout(self.norm1(attention + x))
        
        # The second sub-layer is a feed-forward network
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model for causal language modeling.
    """
    def __init__(
        self,
        vocab_size,
        embed_size=768,
        num_layers=12,
        heads=12,
        ff_hidden_dim=3072,
        dropout=0.1,
        device="cpu",
        max_length=512,
    ):
        super(GPT, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                GPTBlock(embed_size, heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_trg_mask(self, trg):
        """
        Creates a causal mask for the target sequence to prevent attending to future tokens.
        """
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, ids):
        """
        Forward pass for the GPT model.
        The input parameter is named 'ids' to be compatible with the generic train_model function.
        """
        x = ids
        N, seq_length = x.shape
        
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        x = self.dropout((self.embedding(x) + self.position_embedding(positions)))

        # Create the causal mask for self-attention
        mask = self.make_trg_mask(x)

        for layer in self.layers:
            x = layer(x, mask)

        out = self.fc_out(x)
        return out

