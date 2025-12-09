import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Net
from dataload import train_loader
from dataload import test_loader
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# ============ 1. 参数设置 ============
num_epoch = 5
learning_rate = 0.01
save_path = r"mnist_cnn.pth"
data_save_path = r"train_data"

# ============ 2. 损失函数+优化器 ============
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# ============ 3. 初始化 ============
train_losses = []
train_accs = []
test_accs = []

# ============ 4. 训练开始 ============
print("训练开始...")

for epoch in range(num_epoch):
    net.train()
    # 单epoch中数据初始化,用于计算loss和准确率
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # 1. 前向传播
        outputs = net(images)
        loss = criterion(outputs, labels)
        # 2. 反向传播+优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 3. 统计数据
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print("Epoch:{}, Batch:{}/{}, Loss:{}".format(epoch + 1, batch_idx + 1, len(train_loader), loss.item()))

    # 计算本轮训练集的平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print("Epoch:{}训练集：平均损失={}, 准确率={}%".format(epoch + 1, epoch_loss, epoch_acc))

    # ============ 5. 测试/验证 ============
    net.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in test_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    test_accs.append(test_acc)
    print("Epoch:{}测试集：准确率={}%".format(epoch + 1, test_acc))

# ============ 6. 模型保存 ============
torch.save(net.state_dict(), save_path)
print("模型已保存至：{}".format(save_path))

# ============ 7. 可视化 ============
# ============ 修正后的可视化代码 ============
plt.figure(figsize=(12, 8))  # 画布大小保持12*8

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epoch+1), train_losses, 'b-o', label='训练损失')
plt.xlabel('Epoch（训练轮次）')
plt.ylabel('Loss（损失值）')
plt.title('训练损失变化趋势')
plt.legend()       # 显示图例
plt.grid(False)     # 显示网格（半透明，更美观）
plt.xlim(1, num_epoch)        # x轴范围固定为训练轮次

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epoch+1), train_accs, 'r-s', label='训练损失')
plt.plot(range(1, num_epoch+1), train_accs, 'g-^', label='训练损失')
plt.xlabel('Epoch（训练轮次）')
plt.ylabel('Accuracy（准确率，%）')
plt.title('训练/测试准确率变化')
plt.legend()       # 显示图例
plt.grid(False)     # 显示网格（半透明，更美观）
plt.xlim(1, num_epoch)


# 调整子图间距，避免标签重叠
plt.tight_layout(pad=3.0)
# 显示图表
plt.show()
