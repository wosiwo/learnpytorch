import torch
import torch.nn as nn

from torch import tensor

print(torch.__version__)
# 验证 卷积计算 padding 为 same 的情况

# 第一步，我们创建好例子中的（4，4，1）大小的输入特征图
# input_feat = torch.tensor([[4, 1, 7, 5], [4, 4, 2, 5], [7, 7, 2, 4], [1, 0, 2, 4]], dtype=torch.float32)
input_feat = torch.tensor([[4, 1, 7, 5], [4, 4, 2, 5], [7, 7, 2, 4], [1, 0, 2, 4]], dtype=torch.float32).unsqueeze(
    0).unsqueeze(0)
print(input_feat)
print(input_feat.shape)

# 输出：
'''
tensor([[4., 1., 7., 5.],
        [4., 4., 2., 5.],
        [7., 7., 2., 4.],
        [1., 0., 2., 4.]])
torch.Size([4, 4])
'''
# 第二步，创建一个 2x2 的卷积，根据刚才的介绍，输入的通道数为 1，输出的通道数为 1，padding 为’same’

# conv2d = nn.Conv2d(1, 1, (2, 2), stride=1, padding='same', bias=True)
# # 默认情况随机初始化参数
# print(conv2d.weight)
# print(conv2d.bias)
# 输出：
'''
Parameter containing:
tensor([[[[ 0.3235, -0.1593],
          [ 0.2548, -0.1363]]]], requires_grad=True)
Parameter containing:
tensor([0.4890], requires_grad=True)
'''

# 人工干预卷积初始化

conv2d = nn.Conv2d(1, 1, (2, 2), stride=1, padding='same', bias=False)
# 卷积核要有四个维度(输入通道数，输出通道数，高，宽)
kernels = torch.tensor([[[[1, 0], [2, 1]]]], dtype=torch.float32)
conv2d.weight = nn.Parameter(kernels, requires_grad=False)
print(conv2d.weight)
print(conv2d.bias)
# 输出：
'''
Parameter containing:
tensor([[[[1., 0.],
          [2., 1.]]]])
None
'''

output = conv2d(input_feat)
print(output)
