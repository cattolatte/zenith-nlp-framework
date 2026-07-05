import torch
import torch.nn as nn
from .transformer_model import TransformerBlock

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = nn.Embedding(max_len, embed_size)
        self.segment = nn.Embedding(3, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence)
        position = torch.arange(0, sequence.size(1), device=sequence.device).unsqueeze(0)
        x = x + self.position(position)
        x = x + self.segment(segment_label)
        return self.dropout(x)


class BERT(nn.Module):
    def __init__(self, vocab_size, embed_size=768, num_layers=12, heads=12, ff_hidden_dim=3072, dropout=0.1, use_lora=False, lora_rank=4):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        
        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, ff_hidden_dim, dropout, use_lora=use_lora, lora_rank=lora_rank) for _ in range(num_layers)]
        )

        self.mlm_head = nn.Linear(embed_size, vocab_size)

        if use_lora:
            self.freeze_for_lora()

    def freeze_for_lora(self):
        """
        Freezes all parameters of the model except for the LoRA-specific parameters.
        """
        for name, param in self.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        print("Froze base model weights for LoRA fine-tuning.")

    def forward(self, ids, segment_info):
        x = ids
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, segment_info)
        for layer in self.encoder_layers:
            x = layer(x, x, x, mask)
        return x

    def predict_mlm(self, sequence_output):
        return self.mlm_head(sequence_output)
