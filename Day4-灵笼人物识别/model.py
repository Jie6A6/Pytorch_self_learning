import torch
from torch import nn


class VGG19(nn.Module):
    def __init__(self, num_classes=6):
        super(VGG19, self).__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            # 第一模块
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # (3,224,224) -> (64,224,224)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # (64,224,224) -> (64,224,224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64,224,224) -> (64,112,112)
            # 第二模块
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (64,112,112) -> (128,112,112)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # (128,112,112) -> (128,112,112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (128,112,112) -> (128,56,56)
            # 第三模块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (128,56,56) -> (256,56,56)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # (256,56,56) -> (256,56,56)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # (256,56,56) -> (256,56,56)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # (256,56,56) -> (256,56,56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256,56,56) -> (256,28,28)
            # 第四模块
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (256,28,28) -> (512,28,28)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,28,28) -> (512,28,28)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,28,28) -> (512,28,28)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,28,28) -> (512,28,28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (512,28,28) -> (512,14,14)
            # 第五模块
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,14,14) -> (512,14,14)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,14,14) -> (512,14,14)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,14,14) -> (512,14,14)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,14,14) -> (512,14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (512,14,14) -> (512,7,7)
        )
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
        # # 初始化权重
        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

vgg19 = VGG19()
# print(vgg19)

# if __name__ == "__main__":
#     vgg19 = VGG19()
#     input = torch.ones((64, 3, 224, 224))
#     output = vgg19(input)
#     print(output.size())