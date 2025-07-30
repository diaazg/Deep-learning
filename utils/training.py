import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import DataLoader
import torch
import numpy as np
from models.wine_batchNorm import WineModel_BNorm

import torch.nn as nn


def classification_accuracy(y_pred,y):
    matches = torch.argmax(y_pred,axis=1) == y
    matchesNum = matches.float() # convert matches to numeric values
    accuracyPct = 100*torch.mean(matchesNum)
    return accuracyPct




def trainTheM0del(
        doBN=False,
        model=None,
        train_loader=None,
        test_loader=None,
        num_epochs=1000,
        loss_function=nn.BCEWithLogitsLoss(),
        optimizer=None,
        isClassification=False,
        device=torch.device("cpu") 
    ):
    if model is None:
        model = WineModel_BNorm()
    model.to(device)    
    if train_loader is None:
        train_loader = DataLoader([])
    if test_loader is None:
        test_loader = DataLoader([])
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # initialize losses
    losses = torch.zeros(num_epochs)
    train_accuracy = []
    test_accuracy = []

    # loop over epochs
    
    for epoch in range(num_epochs):
        #switch to train mode
        model.train()

        # loop over training data in batches
        batch_accuracy = []
        batch_loss = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            # forward pass and loss
            """  y_pred = model(X,doBN) """
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batch_loss.append(loss.item())

            # compute accuracy for this batch
            if isClassification:
                btchAcc = classification_accuracy(y_pred,y)
                batch_accuracy.append(btchAcc)
            else:
                batch_accuracy.append(100*torch.mean(((y_pred > .5) == y).float()).item())    
        # end of batch loop
        # compute average accuracy and loss for this epoch

        # train_accuracy.append(np.mean(batch_accuracy)) 
        """ This will cause problem in case of not using cpu 
        you/’re trying to use .numpy() or NumPy functions (np.mean) directly on a tensor that’s still on the MPS device,
        and that’s not allowed. NumPy only works with CPU tensors.
        So we need to move tensors to cpu before converting to Numpy or using Numpy functions
        """
        train_accuracy.append(np.mean([acc.cpu().item() if torch.is_tensor(acc) else acc for acc in batch_accuracy]))
        losses[epoch] = np.mean(batch_loss)  # already Python floats, so this is fine

        # test accuracy
        model.eval()
        X,y = next(iter(test_loader)) # extract X,y from test dataloader
        X, y = X.to(device), y.to(device)
        with torch.no_grad(): # deactivates autograd
           """  y_pred = model(X,doBN) """
           y_pred = model(X)
        if isClassification:
            tstAcc = classification_accuracy(y_pred,y)
            tstAcc = tstAcc.cpu().item() if torch.is_tensor(tstAcc) else tstAcc
            test_accuracy.append(tstAcc)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {losses[epoch]:.4f}, Train Acc: {train_accuracy[-1]:.2f}, Test Acc: {test_accuracy[-1]:.2f}')
        else:
            acc = 100 * torch.mean(((y_pred > .5) == y).float()).item()
            test_accuracy.append(acc)   

    return losses, train_accuracy, test_accuracy    