import pandas as pd
from tqdm import tqdm
from utils import preprocessing
from torch.utils.data import DataLoader
from torch.optim import SGD
from model import ForecastingModel

df = pd.read_csv("Data/Output/meals_combined.csv")

preprocessed_data = preprocessing(df)

split = 0.8 

X = preprocessed_data[ : len(preprocessed_data) * 0.8]
y = preprocessed_data[len(preprocessed_data) * 0.8: ]

X_data = DataLoader(X)
y_data = DataLoader(y)

model = ForecastingModel(input_dimensions, hidden_layers, dropout, output_layer)

model.train()
for X_train, y_train in tqdm(zip(data)):
    predictions = model.forward()
    logits # loss function (mse)
    optimizer.backwards()

model.eval()
for X_test, y_test in tqdm(zip(data)):
    predictions = model.forward()
    logits # loss function (mse)

# visulization for your loss going down hopefully
# R^2 coefficient and your final mse 