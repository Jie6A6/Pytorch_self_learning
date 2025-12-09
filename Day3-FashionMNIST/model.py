import torch
from torch import nn


class Fashion(nn.Module):
    def __init__(self):
        super(Fashion, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1),  # (1,28,28) -> (32,26,26)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # (32,26,26) -> (32,13,13)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2),  # (32,13,13) -> (64,6,6)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # (64,6,6) -> (64,3,3)
            nn.ReLU(),
            nn.Flatten(),  # 64*3*3
            nn.Linear(in_features=64 * 3 * 3, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        output = self.model(x)
        return output

# if __name__ == "__main__":
#     fashion = Fashion()
#     input = torch.ones((64, 1, 28, 28))
#     output = fashion(input)
#     print(output.size())