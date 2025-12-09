import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Fashion
from dataload import train_loader
from dataload import test_loader
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题


# ============ 1. 参数设置 ============
fashion = Fashion()
num_epoch = 10
learning_rate = 0.01
save_path = r"fashion_cnn.pth"
data_save_path = r"result_save"


# ============ 2. 优化器 + 损失函数 ============
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fashion.parameters(), lr = learning_rate)


# ============ 3. 数据初始化 ============
train_losses = []
train_accs = []
test_accs = []


# ============ 4. 训练循环 ============
for epoch in range(num_epoch):
    fashion.train()
    # 单轮数据初始化，用于计算Loss和正确率
    running_loss = 0
    correct = 0
    total = 0
    # 模型开始计算
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 1. 前向传播
        outputs = fashion(images)
        loss = criterion(outputs, labels)
        # 2. 反向传播
        optimizer.zero_grad()   # 梯度清零
        loss.backward()         # 反向传播
        optimizer.step()        # 优化参数
        # 3. 数据统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)   # 获取预测类别
        total += labels.size(0)
        correct += (predicted ==labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print("Epoch:{}, Batch:{}/{}, Loss:{}".format(epoch + 1, batch_idx + 1, len(train_loader), loss.item()))

    # ============ 计算本轮的平均损失和准确率 ============
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print("Epoch:{}, 训练集: 平均损失={}, 准确率={}%".format(epoch + 1, epoch_loss, epoch_acc))

    # ============ 测试/验证 ============
    fashion.eval()
    # 单轮数据初始化，用于计算正确率
    test_correct = 0
    test_total = 0
    # 模型开始测试
    with torch.no_grad():   # 禁用梯度计算，减少内存占用
        for images, labels in test_loader:
            outputs = fashion(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = 100 * test_correct / test_total
    test_accs.append(test_acc)
    print("Epoch:{}, 测试集: 准确率={}%".format(epoch + 1, test_acc))


# ============ 5. 模型保存 ============
torch.save(fashion.state_dict(), save_path)
print("模型已保存至: {}".format(save_path))


# # ============ 6. 数据保存 ============
# np.save(
#     data_save_path,
#     {
#         "train_losses": train_losses,
#         "train_accs": train_accs,
#         "test_accs": test_accs,
#         "num_epoch": num_epoch
#     }
# )

# ============ 7.训练过程可视化 ============
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epoch+1), train_losses, 'b', label='训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失变化')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epoch+1), train_accs, 'r', label='训练准确率')
plt.plot(range(1, num_epoch+1), test_accs, 'b', label='训练准确率')
plt.xlabel('Epoch')
plt.ylabel('Accurancy(%)')
plt.title('训练/测试准确率变化')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()