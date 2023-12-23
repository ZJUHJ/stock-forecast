import torch
import torch.nn as nn
import numpy as np


class StockFormer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.8):
        super(StockFormer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=2,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.state_embedding = nn.Embedding(2, input_size)
        # self.pos_embedding = self.sinusoid_encoding_table(15, input_size)
        self.fc = nn.Linear(input_size, num_classes)

    def sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
        
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
        
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table)

    def forward(self, x, state=None):
        tmp = self.state_embedding.weight # (2, input_size)
        state_ids = state.int() # (batch_size, seq_len)
        state_embed = torch.zeros_like(x).to(x.device) # (batch_size, seq_len, input_size)
        state_embed[state_ids==0, :] = tmp[0]
        state_embed[state_ids==1, :] = tmp[1]
        state_embed[:, -1, :] = 0
        # pos_embed = self.pos_embedding[:x.shape[1], :].unsqueeze(0).repeat(x.shape[0], 1, 1)
        # x = x + state_embed + pos_embed
        x = x + state_embed
        x = self.encoder(x)
        x = x[:, -1, :]
        output_logits = self.fc(x)
        return output_logits


class LSTMwClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.8):
        super(LSTMwClassifier, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.encoder(x)
        output_logits = self.fc(out)
        return output_logits

def build_model(input_size=17, hidden_size=64, num_layers=3, num_classes=1):
    # return LSTMwClassifier(input_size, hidden_size, num_layers, num_classes)
    return StockFormer(input_size, hidden_size, num_layers, num_classes)