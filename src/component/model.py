import torch
import torch.nn as nn

class ForecastingModel(nn.Module):
    def __init__(self, model_type: str = "GRU", input_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        m = model_type.upper()
        if m == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
                dropout=0.0 if num_layers == 1 else dropout
            )
        elif m == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
                dropout=0.0 if num_layers == 1 else dropout
            )
        else:
            raise ValueError("MODEL_TYPE must be 'GRU' or 'LSTM'")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]      # (batch, hidden_dim)
        return self.fc(last)      # (batch, 1)