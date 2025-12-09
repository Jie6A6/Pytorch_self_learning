from torch import nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),  # 3*32*32 -> 32*32*32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*15*15

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 32*15*15 -> 64*15*15
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*6*6

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),   # 64*8*8=1024  6-3+1=4
            nn.Linear(in_features=64*8*8, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
        )


    def forward(self, x):
        x = self.sequential(x)
        return x

net = Net()
# print(net)

if __name__ == '__main__':
    net = Net()
    input = torch.ones((64, 3, 32, 32))
    output = net(input)
    print(output.size())