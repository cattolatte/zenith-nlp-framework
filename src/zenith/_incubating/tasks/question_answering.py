import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
from ..core.bpe_tokenizer import BPETokenizer
from ..models.bert_model import BERT
from ..training.training import train_model

class QAModel(nn.Module):
    def __init__(self, backbone, embed_size):
        super(QAModel, self).__init__()
        self.backbone = backbone
        self.qa_outputs = nn.Linear(embed_size, 2)
        
    def forward(self, ids, segment_info):
        features = self.backbone(ids, segment_info)
        logits = self.qa_outputs(features)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        context = str(row.text)
        question = "What is the main topic?"
        context_tokens = self.tokenizer.tokenize(context)
        if len(context_tokens) < 10:
            context_tokens += [0] * (10 - len(context_tokens))

        answer_start_idx = torch.randint(0, len(context_tokens) - 5, (1,)).item()
        answer_end_idx = answer_start_idx + 5

        question_tokens = self.tokenizer.tokenize(question)
        
        input_tokens = [1] + question_tokens + [2] + context_tokens + [2]
        segment_ids = [0] * (len(question_tokens) + 2) + [1] * (len(context_tokens) + 1)

        final_answer_start = answer_start_idx + len(question_tokens) + 2
        final_answer_end = answer_end_idx + len(question_tokens) + 2

        if len(input_tokens) < self.max_len:
            pad_len = self.max_len - len(input_tokens)
            input_tokens += [0] * pad_len
            segment_ids += [0] * pad_len
        else:
            input_tokens = input_tokens[:self.max_len]
            segment_ids = segment_ids[:self.max_len]
            if final_answer_start >= self.max_len: final_answer_start = self.max_len - 1
            if final_answer_end >= self.max_len: final_answer_end = self.max_len - 1

        return {
            "ids": torch.tensor(input_tokens, dtype=torch.long),
            "segment_info": torch.tensor(segment_ids, dtype=torch.long),
            "start_positions": torch.tensor(final_answer_start, dtype=torch.long),
            "end_positions": torch.tensor(final_answer_end, dtype=torch.long)
        }

if __name__ == '__main__':

    print("This script is ready to be used with the centralized training function.")
    print("You would need to instantiate the dataset and model, then call train_model.")

