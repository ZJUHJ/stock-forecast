import torch
import torch.nn as nn

class StockFormer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.8):
        super(StockFormer, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_size,
            nhead=8,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.state_embedding = nn.Embedding(2, input_size)

    def forward(self, x):
        return None
