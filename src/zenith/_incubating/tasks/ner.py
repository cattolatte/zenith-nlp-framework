import torch
import torch.nn as nn
from torch.utils.data import random_split
import pandas as pd
from functools import partial

from ..data.loaders import GenericNLPDataset
from ..core.bpe_tokenizer import BPETokenizer
from ..models.bert_model import BERT
from ..training.training import train_model

class NERModel(nn.Module):
    def __init__(self, backbone, embed_size, num_classes):
        super(NERModel, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(embed_size, num_classes)
        
    def forward(self, ids, segment_info=None):
        if isinstance(self.backbone, BERT):
            features = self.backbone(ids, segment_info)
        else:
            features = self.backbone(ids)
        return self.fc(features)

def process_ner_data(row, tokenizer, num_classes, max_len):
    """Processor function for Named Entity Recognition."""
    text = str(row.text)
    token_ids = tokenizer.tokenize(text)
    labels = torch.randint(0, num_classes, (len(token_ids),)).tolist()
    
    if len(token_ids) < max_len:
        pad_len = max_len - len(token_ids)
        token_ids += [0] * pad_len
        labels += [-100] * pad_len  # Use -100 to ignore padding in loss
    else:
        token_ids = token_ids[:max_len]
        labels = labels[:max_len]
        
    return {
        "ids": torch.tensor(token_ids, dtype=torch.long),
        "segment_info": torch.ones(max_len, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

if __name__ == '__main__':
    DATA_PATH = "data/text_dataset_20240717141623.csv"
    VOCAB_SIZE = 10000
    NUM_NER_CLASSES = 5 
    EPOCHS = 3
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    MAX_LEN = 128

    df = pd.read_csv(DATA_PATH).dropna().sample(n=500, random_state=42)

    try:
        tokenizer = BPETokenizer()
        tokenizer.load("bpe_tokenizer.json")
    except FileNotFoundError:
        print("Training BPE tokenizer...")
        tokenizer = BPETokenizer()
        tokenizer.train(df['text'].tolist(), VOCAB_SIZE)
        tokenizer.save("bpe_tokenizer.json")

    processor = partial(process_ner_data, tokenizer=tokenizer, num_classes=NUM_NER_CLASSES, max_len=MAX_LEN)
    dataset = GenericNLPDataset(df, processor)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    bert_backbone = BERT(vocab_size=len(tokenizer.token_to_id))
    model = NERModel(backbone=bert_backbone, embed_size=768, num_classes=NUM_NER_CLASSES)

    print("Starting NER model training...")
    train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        save_path="ner_model.pth"
    )
    print("NER training complete.")

