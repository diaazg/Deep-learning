import wine_preprocessing as wp
# for DL modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# for number-crunching
import numpy as np
import scipy.stats as stats

# for dataset management
import pandas as pd



def load_wine_data():
    wine_data = wp.load_wine_data()
    wine_data = wp.preprocess_wine_data(wine_data)
    return wine_data


def convert_to_tensor(wine_data):
    wine_df = wine_data.copy()
    label = 'boolQuality'
    features = wine_df.drop(columns=['quality', 'boolQuality']).columns
    dataTensor = torch.tensor(wine_df[features].values, dtype=torch.float32)
    labelTensor = torch.tensor(wine_df[label].values, dtype=torch.float32)
    labelTensor = labelTensor[:, None]

    return dataTensor, labelTensor

def split_data(dataTensor, labelTensor):
    X_train, X_test, y_train, y_test = train_test_split(dataTensor, labelTensor, test_size=0.2, random_state=42)
    # then convert them into PyTorch datasets (note: already converted to tensors)
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    return train_data, test_data

def create_dataloaders(train_data, test_data,train_batch_size,test_batch_size):
    train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size, shuffle=True)
    return train_loader, test_loader





def reorgnize_dataloaders(train_batch_size=32,test_batch_size=32):
    wine_data = load_wine_data()
    dataTensor, labelTensor = convert_to_tensor(wine_data)
    print(dataTensor.shape)
    train_data, test_data = split_data(dataTensor, labelTensor)
    train_loader, test_loader = create_dataloaders(train_data, test_data,train_batch_size,test_batch_size)
    return train_loader, test_loader



