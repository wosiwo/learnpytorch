import torch
import torch.nn as nn

# 2个神经元的输出y的数值为
y = torch.randn(2)
print(y)

'''
输出：tensor([0.2370, 1.7276])
'''
m = nn.Softmax(dim=0)
out = m(y)
print(out)
'''
输出：tensor([0.1838, 0.8162])
'''
print("=================")

x = torch.randint(0, 255, (1, 128*128), dtype=torch.float32)
fc = nn.Linear(128*128, 2)
y = fc(x)
print(y)
'''
输出：tensor([[  72.1361, -120.3565]], grad_fn=<AddmmBackward>)
'''
# 注意y的shape是(1, 2)
output = nn.Softmax(dim=1)(y)
print(output)
'''
输出：tensor([[1., 0.]], grad_fn=<SoftmaxBackward>)
'''