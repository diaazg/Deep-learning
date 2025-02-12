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
        isClassification=False
    ):
    if model is None:
        model = WineModel_BNorm()
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
            # forward pass and loss
            y_pred = model(X,doBN)
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
        train_accuracy.append(np.mean(batch_accuracy))
        losses[epoch] = np.mean(batch_loss)

        # test accuracy
        model.eval()
        X,y = next(iter(test_loader)) # extract X,y from test dataloader
        with torch.no_grad(): # deactivates autograd
            y_pred = model(X,doBN)
        if isClassification:
            tstAcc = classification_accuracy(y_pred,y)
            test_accuracy.append(tstAcc)
        else:
            test_accuracy.append(100*torch.mean(((y_pred > .5) == y).float()).item())    

    return losses, train_accuracy, test_accuracy    