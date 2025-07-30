import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss


def trainAE(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 1000,
    loss_function:_Loss = nn.MSELoss(),
    device: torch.device = torch.device("cpu")
):
    
    train_loss = torch.zeros(num_epochs)
    test_loss = torch.zeros(num_epochs)

    for epoch in range(num_epochs):
        model.train()
        batch_loss = []
        for X,_ in train_loader:
            X = X.to(device)

            y_pred = model(X)
            loss: torch.Tensor = loss_function(y_pred, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        train_loss[epoch] = np.mean(batch_loss)    

        model.eval()

        X,_ = next(iter(test_loader))
        X = X.to(device)

        with torch.no_grad():
            y_pred = model(X)
            loss: torch.Tensor = loss_function(y_pred, X)

        test_loss[epoch] = loss.item()   
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss[epoch]:.4f} - Test Loss: {test_loss[epoch]:.4f}")

    return train_loss,test_loss,model         

   