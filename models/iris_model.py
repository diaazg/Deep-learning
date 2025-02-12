import torch.nn as nn
import torch.nn.functional as F
import torch

class IrisModel(nn.Module):
    def __init__(self, hiddenFun='relu'):
        super().__init__()
        self.hiddenFun = hiddenFun

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.output = nn.Linear(10, 3)

    def forward(self, x, doBN=False):
        
        activation_function = getattr(F, self.hiddenFun, None)
        if activation_function is None:
            raise ValueError(f"Activation function '{self.hiddenFun}' is not supported.")

        
        x = activation_function(self.fc1(x))
        
        
        x = activation_function(self.fc2(x))
        
        
        x = self.output(x)
        return F.softmax(x, dim=1)
