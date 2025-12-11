import numpy as np
import pandas as pd # type: ignore
import torch # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore



# ============ 1. 基础配置 ============
stock_code = 'AAPL'
time_step = 60
batch_size = 32



# ============ 2. 数据加载+数据划分 ============
data = pd.read_csv(r'dataset\SH600519.csv')   # 读取股票文件数据
# print(data)

# 前2125天的开盘价作为训练集，后300天的开盘价作为测试集
train_set = data.iloc[0:2426-300, 2:3].values   # iloc[]位置索引,左闭右开,.values转换为numpy数组
test_set = data.iloc[2426-300:, 2:3].values
# print(train_set)
# print(test_set)

# 归一化
scaler = MinMaxScaler(feature_range = (0, 1))
train_set = scaler.fit_transform(train_set)
test_set = scaler.transform(test_set)   # 训练集和测试集不同，训练集的缩放参数（min/max）要复用在测试集上
# print(train_set)
# print(test_set)


# ============ 3. 数据划分 ============
# 设置训练集测试集
x_train = []    # 存储训练集输入（每组是60天的开盘价）
y_train = []    # 存储训练集标签（每组对应第61天的开盘价）
x_test = []     # 存储测试集输入（每组是60天的开盘价）
y_test = []     # 存储训练集标签（每组对应第61天的开盘价）
'''
前60天的开盘价作为输入特征x_train
第61天的开盘价作为输入标签y_train
for循环共构建2426-300-60=2066组训练数据。
       共构建300-60=260组测试数据
'''
for i in range(60, len(train_set)):
    x_train.append(train_set[i - 60:i, 0])  # 行切片:截取第i-60行(包含)到第i行(不包含)的所有行,只截取第一列的元素
    y_train.append(train_set[i, 0])
for i in range(60, len(test_set)):
    x_test.append(test_set[i - 60:i, 0])  # 行切片:截取第i-60行(包含)到第i行(不包含)的所有行,只截取第一列的元素
    y_test.append(test_set[i, 0])

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

# 对数据集打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)  # 同样的随机种子是为了让样本和标签对应
# print(len(x_train))
# print(x_train)


"""
将训练数据调整为数组（array）
调整后的形状：
x_train:(2066, 60, 1)
y_train:(2066,)
x_test :(240, 60, 1)
y_test :(240,)
"""
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)
# print(x_train.shape)
# print(x_train.size)
# print(x_train)
"""
输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
"""
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
x_ttest = np.reshape(x_test, (x_test.shape[0], 60, 1))