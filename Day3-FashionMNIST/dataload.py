import torch
from torchvision import datasets
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 数据预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])


# 2. 训练集和数据集下载
train_dataset = datasets.FashionMNIST(
    root = r"data",
    train = True,
    download = True,
    transform = transform
)

test_dataset = datasets.FashionMNIST(
    root = r"data",
    train = False,
    download = True,
    transform = transform
)

# 3. DataLoader批量加载数据
train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 64,
    shuffle = False
)

# # 4. 验证数据加载结果
# print("训练集样本数：{}".format(len(train_dataset)))
# print("测试集样本数：{}".format(len(test_dataset)))
# print("单个样本形状：{}".format(train_dataset[0][0].shape))
# print("单个样本标签：{}".format(train_dataset[0][1]))

# # 5. 可视化数据
# images, labels = next(iter(train_loader))

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure(figsize=(20,10))
# for i in range(20):
#     img = images[i].numpy().squeeze()
#     plt.subplot(5,10,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(img, cmap=plt.cm.binary)
#     plt.xlabel(class_names[labels[i].item()])
# plt.show()