import torch.nn as nn
import torch

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()

        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 3)

    def forward(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x




