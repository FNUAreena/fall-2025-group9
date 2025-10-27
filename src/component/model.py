#%%
import torch
import torch.nn as nn

class ForecastingModel(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(ForecastingModel, self).__init__()
        self.model_type = model_type.upper()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        if self.model_type == 'NAIVE':
            self.is_naive = True
            return

        if self.model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        elif self.model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        else:
            raise ValueError("model_type must be 'LSTM' or 'GRU'")

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if getattr(self, "is_naive", False):
            return x[:, -1, 0].unsqueeze(1)
       
        out, _ = self.rnn(x)
        out = out[:, -1, :]  
        out = self.fc(out)
        return out
