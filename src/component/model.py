#%%
from torch.nn import LSTM, Linear, Sequential

class ForecastingModel(torch.nn):
    def __init__(self):
        self.model = Sequential([
            LSTM(),
            Linear()
        ])
    
    def forward(self, x):
        return self.model(x)
