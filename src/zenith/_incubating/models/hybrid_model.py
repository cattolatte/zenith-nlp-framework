import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.sequence_model import SequenceModel

class Hybrid_CNN_RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, cnn_out_channels, cnn_kernel_size, 
                 rnn_hidden_dim, rnn_num_layers, output_dim, dropout):
        """
        Initializes the Hybrid CNN-RNN model.

        Args:
            vocab_size (int): The size of the vocabulary.
            embed_size (int): The dimensionality of the word embeddings.
            cnn_out_channels (int): The number of output channels for the CNN layer.
            cnn_kernel_size (int): The kernel size for the CNN layer.
            rnn_hidden_dim (int): The number of features in the hidden state of the RNN.
            rnn_num_layers (int): The number of recurrent layers in the RNN.
            output_dim (int): The dimension of the output layer.
            dropout (float): The dropout rate.
        """
        super(Hybrid_CNN_RNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 1D Convolutional layer
        # Padding is set to keep the sequence length the same
        self.conv = nn.Conv1d(
            in_channels=embed_size, 
            out_channels=cnn_out_channels, 
            kernel_size=cnn_kernel_size, 
            padding=(cnn_kernel_size - 1) // 2
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Use the existing SequenceModel (LSTM)
        # The input dimension for the RNN is the number of output channels from the CNN
        self.rnn = SequenceModel(
            input_dim=cnn_out_channels,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            output_dim=output_dim
        )

    def forward(self, x):
        """
        Forward pass for the Hybrid_CNN_RNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # 1. Embedding layer
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embed_size)
        
        # 2. Permute for Conv1d
        # Conv1d expects (batch_size, embed_size, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # 3. Convolutional layer
        conved = self.conv(embedded)
        # conved shape: (batch_size, cnn_out_channels, seq_len)
        conved = F.relu(conved)
        
        # 4. Permute back for RNN
        # RNN expects (batch_size, seq_len, features)
        conved = conved.permute(0, 2, 1)
        
        # Apply dropout
        conved = self.dropout(conved)
        
        # 5. Pass through RNN
        # The existing SequenceModel will handle the LSTM and final linear layer
        output = self.rnn(conved)
        
        return output