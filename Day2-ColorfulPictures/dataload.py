import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# ================1. 数据预处理变换================
# 训练集：数据增强+标准化
train_transform = transforms.Compose([
    # 随机裁剪（填充4像素后裁剪32×32）
    transforms.RandomCrop(32, padding=4),
    # 随机水平翻转（概率为0.5）
    transforms.RandomHorizontalFlip(),
    # 转换为张量
    transforms.ToTensor(),
    # 标准化
    # transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2470, 0.2435, 0.2616]
    # )
])

# 测试集：数据增强+标准化
test_transform = transforms.Compose([
    # 转换为张量
    transforms.ToTensor(),
    # 标准化
    # transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2470, 0.2435, 0.2616]
    # )
])

# ================2. 加载数据集================
# 训练集
train_dataset = torchvision.datasets.CIFAR10(
    root=r'data',
    train=True,
    download=True,
    transform=train_transform
)

# 测试集
test_dataset = torchvision.datasets.CIFAR10(
    root=r'data',
    train=False,
    download=True,
    transform=test_transform
)


# ================3. 创建数据加载器================
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

# ================4. 验证数据集================
print("训练的样本数{}".format(len(train_loader)))
print("测试的样本数{}".format(len(test_loader)))
print("图形形状：{}".format(train_dataset[0][0].shape))
print("类别：{}".format(train_dataset[0][1]))


# ================5. 可视化展示部分图片================
class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

plt.figure(figsize=(20, 10))
for i in  range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_dataset[i][0].numpy().transpose(1, 2, 0))
    plt.xlabel(class_names[train_dataset[i][1]])
plt.show()