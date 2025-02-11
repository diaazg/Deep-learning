import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from models.wine_batchNorm import WineModel_BNorm

import torch.nn as nn


def trainTheM0del(doBN=True,model=WineModel_BNorm(),train_loader=[],test_loader=[],num_epochs=1000):

    # loss function and optimizer
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)

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
        test_accuracy.append(100*torch.mean(((y_pred > .5) == y).float()).item())

    return losses, train_accuracy, test_accuracy    