# src/my_nlp_framework/core/sequence_model.py
import torch
import torch.nn as nn

class SequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(SequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out
