import time

import numpy as np
import random
from matplotlib import pyplot as plt


# 线性回归模型训练
def makeTrainData():
    w = 2
    b = 3
    xlim = [-10, 10]
    x_train = np.random.randint(low=xlim[0], high=xlim[1], size=30)

    y_train = [w * x + b + random.randint(0, 2) for x in x_train]

    # plt.plot(x_train, y_train, 'bo')
    # plt.show()
    return x_train,y_train

x_train, y_train = makeTrainData()
print(x_train)
print(y_train)

import torch
from torch import nn

class LinearModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(1))
    self.bias = nn.Parameter(torch.randn(1))

    for parameter in self.named_parameters():
        print(parameter)

  def forward(self, input):
    return (input * self.weight) + self.bias


model = LinearModel()
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

y_train = torch.tensor(y_train, dtype=torch.float32)
for _ in range(1000):
    input = torch.from_numpy(x_train)
    output = model(input)
    loss = nn.MSELoss()(output, y_train)
    model.zero_grad()
    loss.backward()
    optimizer.step()

# 查看参数
for parameter in model.named_parameters():
  print(parameter)
# 输出：
'''
('weight', Parameter containing:
tensor([2.0071], requires_grad=True))
('bias', Parameter containing:
tensor([3.1690], requires_grad=True))
'''

# state_dict 存储的是模型可训练的参数
print(model.state_dict())


# 模型的保存与加载

# 保存整个模型
print("model save \n")
torch.save(model, './linear_model_with_arc.pth')
# 加载模型，不需要创建网络了
print("model load \n")
linear_model_2 = torch.load('./linear_model_with_arc.pth')
linear_model_2.eval()
for parameter in linear_model_2.named_parameters():
  print(parameter)
# 输出：
'''
('weight', Parameter containing:
tensor([[2.0071]], requires_grad=True))
('bias', Parameter containing:
tensor([3.1690], requires_grad=True))
'''


