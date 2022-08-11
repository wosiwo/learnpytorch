# 使用苹果m1芯片加速计算
import torch
import platform
print(torch.__version__)

# 查看是否arm版python
print(platform.uname()[4])
# 查看是pytorch否支持mps
print(torch.backends.mps.is_built())

foo = torch.rand(1, 3, 224, 224).to('mps')

device = torch.device('mps')
foo = foo.to(device)
print(foo)
'''
Pytorch已经支持下面这些device了，确实出乎意料: 
cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, ort, mps, xla, lazy, vulkan, meta, hpu
'''


