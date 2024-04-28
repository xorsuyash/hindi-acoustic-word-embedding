import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

class SimpleLSTM(nn.Module):
    def __init__(self, config):
        super(SimpleLSTM, self).__init__()

        hidden_size = config['hidden_size']
        view1_input_size = config['view1_input_size']
        view2_input_size = config['view2_input_size']
        margin = config['margin']
        lr = config['learning_rate']
        kp = config['keep_prob']
        obj = config['objective']

        # LSTM layers
        self.lstm_layer1_view1 = nn.LSTM(view1_input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_layer2_view1 = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_layer1_view2 = nn.LSTM(view2_input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_layer2_view2 = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True, bidirectional=True)

        # Normalization layer
        self.norm = nn.LayerNorm(2 * hidden_size)

        self.margin = margin
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x1, x1_lengths, x2=None, x2_lengths=None, c1=None, c2=None):
        # View 1 Layer 1
        x1_packed = nn.utils.rnn.pack_padded_sequence(x1, x1_lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_layer1_out, _ = self.lstm_layer1_view1(x1_packed)
        lstm_layer1_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_layer1_out, batch_first=True)

        # View 1 Layer 2
        lstm_layer2_out, _ = self.lstm_layer2_view1(lstm_layer1_out)

        x1_indices = torch.arange(x1.size(0)) * x1.size(1) + (x1_lengths - 1)
        x1_final = torch.index_select(lstm_layer2_out.reshape(-1, lstm_layer2_out.size(2)), 0, x1_indices)

        if x2 is not None:
            # View 2 Layer 1
            x2_packed = nn.utils.rnn.pack_padded_sequence(x2, x2_lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_layer1_out, _ = self.lstm_layer1_view2(x2_packed)
            lstm_layer1_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_layer1_out, batch_first=True)

            # View 2 Layer 2
            lstm_layer2_out, _ = self.lstm_layer2_view2(lstm_layer1_out)

            x2_indices = torch.arange(x2.size(0)) * x2.size(1) + (x2_lengths - 1)
            x2_final = torch.index_select(lstm_layer2_out.reshape(-1, lstm_layer2_out.size(2)), 0, x2_indices)

        if c1 is not None:
            # View 2 Layer 1 for c1
            c1_packed = nn.utils.rnn.pack_padded_sequence(c1, x2_lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_layer1_out, _ = self.lstm_layer1_view2(c1_packed)
            lstm_layer1_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_layer1_out, batch_first=True)

            # View 2 Layer 2 for c1
            lstm_layer2_out, _ = self.lstm_layer2_view2(lstm_layer1_out)

            c1_indices = torch.arange(c1.size(0)) * c1.size(1) + (x2_lengths - 1)
            c1_final = torch.index_select(lstm_layer2_out.reshape(-1, lstm_layer2_out.size(2)), 0, c1_indices)

        if c2 is not None:
            # View 1 Layer 1 for c2
            c2_packed = nn.utils.rnn.pack_padded_sequence(c2, x1_lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_layer1_out, _ = self.lstm_layer1_view1(c2_packed)
            lstm_layer1_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_layer1_out, batch_first=True)

            # View 1 Layer 2 for c2
            lstm_layer2_out, _ = self.lstm_layer2_view1(lstm_layer1_out)

            c2_indices = torch.arange(c2.size(0)) * c2.size(1) + (x1_lengths - 1)
            c2_final = torch.index_select(lstm_layer2_out.reshape(-1, lstm_layer2_out.size(2)), 0, c2_indices)

        if c1 is not None and c2 is not None:
            loss = self.contrastive_loss(self.margin, x1_final, c1_final, c2_final)
        elif c1 is not None:
            loss = self.contrastive_loss(self.margin, x1_final, c1_final, x2_final)
        elif c2 is not None:
            loss = self.contrastive_loss(self.margin, c1_final, x1_final, c2_final)
        else:
            loss = self.contrastive_loss(self.margin, c1_final, x1_final, x2_final)
        return loss

    def contrastive_loss(self, margin, x1, c1, c2):
        sim = torch.mul(x1, c1)
        sim = torch.sum(sim, dim=1)
        dis = torch.mul(x1, c2)
        dis = torch.sum(dis, dim=1)
        return torch.mean(torch.max(margin + dis - sim, torch.tensor(0.0)))

    def normalization(self, x):
        return nn.functional.normalize(x, p=2, dim=1)


if __name__=='__main__':

 

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    def estimate_model_size(model):
        num_params = count_parameters(model)
        size_bytes = num_params * 4  # Assuming float32 data type
        size_mb = size_bytes / (1024 * 1024)
        return size_mb

    # Create an instance of your SimpleLSTM model
    config = {
        'hidden_size': 512,
        'view1_input_size': 100,
        'view2_input_size': 150,
        'margin': 1.0,
        'learning_rate': 0.001,
        'keep_prob': 0.8,
        'objective': [0, 1, 2, 3]  # Example objectives
    }
    model = SimpleLSTM(config)

    # Calculate and print the estimated size of the model
    size_mb = estimate_model_size(model)
    print(f"Estimated model size: {size_mb:.2f} MB")


