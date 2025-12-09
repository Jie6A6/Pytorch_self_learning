import torch
from torchvision import datasets
import torchvision  # type: ignore
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 数据预处理
# PIL -> Tensor
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), # PIL->Tensor,像素[0,255]->[0,1]
    torchvision.transforms.Normalize((0.1307,), (0.3081,)) # 官方推荐的均值和标准差
])

# 2. 下载和加载训练集和测试集
train_dataset = datasets.MNIST(
    root=r"data",
    train = True,
    download=True,
    transform = transform
)

test_dataset = datasets.MNIST(
    root=r"data",
    train = False,
    download=True,
    transform = transform
)

# 3. DataLoader批量加载数据
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)

# # 4. 验证数据加载结果
# print("训练集样本数：{}".format(len(train_dataset)))
# print("测试集样本数：{}".format(len(test_dataset)))
# print("单个样本形状：{}".format(train_dataset[0][0].shape))
# print("单个样本标签：{}".format(train_dataset[0][1]))
#
# # 5. 可视化数据
# images, labels = next(iter(train_loader))
#
# plt.figure(figsize=(12, 6))
#
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     img = images[i].numpy().squeeze()   # 移除张量 / 数组中所有维度为 1 的轴（维度）
#     plt.imshow(img, cmap='gray')
#     plt.title("Label:{}".format(labels[i].item()))
#     plt.axis('off')
#
# plt.tight_layout()
# plt.show()