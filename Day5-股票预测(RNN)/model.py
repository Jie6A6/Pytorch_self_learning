import torch  # type: ignore
from torch import nn  # type: ignore


class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=80, output_size=1):
        super(SimpleRNNModel, self).__init__()
        # 第一层RNN: return_sequences=True(返回所有时间步的输出)
        self.rnn1 = nn.RNN(
            input_size=input_size,  # 输入特征维度：股票开盘价是1维（只传一个数值）
            hidden_size=hidden_size,  # 隐藏层维度：80个“记忆单元”，决定记忆容量
            num_layers=1,  # 单层RNN(不是堆叠)
            batch_first=True  # 输入形状: (批次大小, 序列长度, 输入维度)
        )
        self.dropout1 = nn.Dropout(p=0.2)  # 随机让20%的神经元输出为0, 防止过拟合
        # 第二层RNN: return_sequences=False(返回所有时间步的输出)
        self.rnn2 = nn.RNN(
            input_size=hidden_size,  # 输入维度=上层输出维度(80)
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(p=0.2)  # 随机让20%的神经元输出为0, 防止过拟合
        # 输出层: 把80维的隐藏状态 -> 1维度(预测开盘价)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 第一步: 第一层RNN(解包，只取输出out，忽略隐藏状态单层RNN时就是(1, batch_size, 80))
        out, _ = self.rnn1(x)  # out形状: (batch_size, seq_len, hidden_size)
        out = self.dropout1(out)

        # 第二步: 第二层RNN(解包，只取输出out，忽略隐藏状态单层RNN时就是(1, batch_size, 80))
        out, _ = self.rnn2(out)  # out形状: (batch_size, seq_len, hidden_size)
        out = self.dropout2(out)

        # 第三步: 取最后一个时间步的输出
        out = out[:, -1, :]  # 切片后形状：(batch_size, hidden_size)

        # 第四步: 全连接层输出预测值
        out = self.fc(out)
        return out


model = SimpleRNNModel()
print(model)