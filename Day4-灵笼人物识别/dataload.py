import pathlib

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# ============ 1. 随机种子设置 ============
np.random.seed(0)
torch.manual_seed(0)

# ============ 2. 数据路径 ============
data_dir = r'D:\OneDrive\04-Python\Pytorch_self_learning\Day4-灵笼人物识别\datasets\linglong_photos'
# 让路径操作从 “字符串拼接 / 切割” 的繁琐方式，变成 “面向对象” 的简洁操作。
data_dir = pathlib.Path(data_dir)

# ============ 3. 查看数据基本信息 ============
image_count = len(list(data_dir.glob('*/*')))
print('图片总数:{}'.format(image_count))

# ============ 4. 数据预处理 ============
# 定义数据变换
transform = torchvision.transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# ============ 5. 加载数据集 ============
full_dataset = torchvision.datasets.ImageFolder(
    root=data_dir,
    transform=transform
)
# ============ 6. 划分训练集和测试集(9:1) ============
val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset,   # 待拆分的完整数据集
    [train_size, val_size], # 训练集/验证集的样本数列表
    generator=torch.Generator().manual_seed(0)  # # 专属随机生成器
)
# ============ 7. 创建数据加载器 ============
batch_size = 16
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)
# ============ 8. 获取类别名称 ============
class_names = full_dataset.classes
print('类别名称:{}'.format(class_names))
print('训练集数量:{}'.format(len(train_dataset)))
print('验证集数量:{}'.format(len(val_dataset)))
# ============ 9. 可视化数据 ============
# 反归一化，用于可视化
def denormalize(tensor):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    tensor = tensor * std + mean
    return tensor
# 可视化训练集样本
plt.figure(figsize=(10, 5))
images, labels = next(iter(train_loader))
for i in range(8):
    plt.subplot(2, 8, i + 1)
    # 反归一化并转换为numpy数组
    img = denormalize(images[i]).numpy().transpose(1, 2, 0) # [C,H,W] -> [H,W,C]
    img = np.clip(img, 0, 1)
    plt.imshow(img) # imshow需要[H,W,C],Tensor是[C,H,W]
    plt.title(class_names[labels[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()
# 检查数据形状
print('图像批次形状:{}'.format(images.shape))
print('标签批次形状:{}'.format(labels.shape))