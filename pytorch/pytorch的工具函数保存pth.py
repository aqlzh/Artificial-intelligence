import torch
import torch.nn as nn

# 1、虚拟测试数据
x = torch.randn(2, 3)
print(x)
print(nn.functional.relu(x))
print(nn.functional.sigmoid(x))
print(nn.functional.softmax(x, dim=1))
print(nn.functional.tanh(x))

# 2、Dropout层，特征归零，缓解神经网络过拟合
dropout = torch.nn.Dropout(p=0.5)
print(dropout(x))

# 3、保存模型的参数
model = torch.nn.Linear(3, 2)

torch.save(model.state_dict(), 'temp.pth')  # 保存网络参数
model.load_state_dict(torch.load('temp.pth'))

# 4、手动初始化参数
