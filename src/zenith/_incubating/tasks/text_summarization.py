import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
from ..core.bpe_tokenizer import BPETokenizer
from ..models.transformer_model import Seq2SeqTransformer
from ..training.training import train_model

class SummarizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_source_len, max_target_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        
        self.pad_token_id = tokenizer.token_to_id.get('<pad>', 0)
        self.sos_token_id = tokenizer.token_to_id.get('<sos>', 1)
        self.eos_token_id = tokenizer.token_to_id.get('<eos>', 2)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        source_text = str(row.text)
        summary_text = ' '.join(source_text.split()[:15])

        source_tokens = self.tokenizer.tokenize(source_text)
        target_tokens = self.tokenizer.tokenize(summary_text)

        source_tokens = source_tokens[:self.max_source_len - 2]
        target_tokens = target_tokens[:self.max_target_len - 1]

        src_tensor = [self.sos_token_id] + source_tokens + [self.eos_token_id]
        src_padding = [self.pad_token_id] * (self.max_source_len - len(src_tensor))
        src_tensor.extend(src_padding)

        trg_tensor_input = [self.sos_token_id] + target_tokens
        trg_padding_input = [self.pad_token_id] * (self.max_target_len - len(trg_tensor_input))
        trg_tensor_input.extend(trg_padding_input)

        trg_tensor_label = target_tokens + [self.eos_token_id]
        trg_padding_label = [-100] * (self.max_target_len - len(trg_tensor_label))
        trg_tensor_label.extend(trg_padding_label)
        
        return {
            "src": torch.tensor(src_tensor, dtype=torch.long),
            "trg": torch.tensor(trg_tensor_input, dtype=torch.long),
            "label": torch.tensor(trg_tensor_label, dtype=torch.long)
        }

if __name__ == '__main__':
    DATA_PATH = "data/text_dataset_20240717141623.csv"
    VOCAB_SIZE = 10000
    EPOCHS = 5
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 8
    MAX_SOURCE_LEN = 200
    MAX_TARGET_LEN = 50
    
    df = pd.read_csv(DATA_PATH).dropna().sample(n=500, random_state=42)
    
    try:
        tokenizer = BPETokenizer()
        tokenizer.load("bpe_tokenizer.json")
    except FileNotFoundError:
        print("Training BPE tokenizer...")
        tokenizer = BPETokenizer()
        special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        tokenizer.token_to_id = special_tokens
        tokenizer.id_to_token = {v: k for k, v in special_tokens.items()}
        corpus = df['text'].tolist()
        tokenizer.train(corpus, VOCAB_SIZE)
        tokenizer.save("bpe_tokenizer.json")

    vocab_size = len(tokenizer.token_to_id)
    PAD_IDX = tokenizer.token_to_id['<pad>']

    dataset = SummarizationDataset(df, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = Seq2SeqTransformer(
        src_vocab_size=vocab_size,
        trg_vocab_size=vocab_size,
        src_pad_idx=PAD_IDX,
        trg_pad_idx=PAD_IDX
    )

    print("Starting summarization model training with centralized training function...")
    train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        save_path="summarization_model.pth"
    )
    print("Summarization training complete.")

