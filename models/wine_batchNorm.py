import torch.nn as nn
import torch.nn.functional as F

class WineModel_BNorm(nn.Module):

    def __init__(self):
        super().__init__()

        # input layer
        self.input = nn.Linear(11, 16)

        # hidden layers
        self.fc1 = nn.Linear(16, 32) # fc means fully connected
        self.bnorm1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(32, 20)
        self.bnorm2 = nn.BatchNorm1d(32)

        # output layer
        self.output = nn.Linear(20, 1)

    # forward pass
    def forward(self, x,doBN):
      
      x = F.relu(self.input(x)) # no batch normalization for input layer

      if doBN:
        # hidden layer 1
        x = self.bnorm1(x) # batch normalization
        x = self.fc1(x) # weighted combination
        x = F.relu(x) # activation function

        # hidden layer 2

        x = self.bnorm2(x) 
        x = self.fc2(x)
        x = F.relu(x)

      else:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  

      return self.output(x)  