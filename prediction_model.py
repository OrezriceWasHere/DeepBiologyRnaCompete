import torch.nn as nn
import torch.nn.functional as F


class PredictionModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=args.one_hot_size, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(200, 1)

        # self.model = torch.nn.Sequential(
        #     self.conv1,
        #     nn.SELU(),
        #     self.pool1,
        #     nn.Linear(20, 1)
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = F.selu(x)
        x = self.pool1(x)
        x = x.view(-1, 200)
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x
