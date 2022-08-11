import torchvision.models as models
import torch
from torch import nn

# 加载模型(本地没有则从网上下载)
from torch.utils.data import DataLoader

alexnet = models.alexnet(pretrained=True)

# 使用模型预测狗狗类型
from PIL import Image
import torchvision
import torchvision.transforms as transforms

im = Image.open('./images/dog.webp')

transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

input_tensor = transform(im).unsqueeze(0)
# alexnet(input_tensor).argmax()
print(alexnet(input_tensor).argmax())
'''
输出：263
运行了前面的代码之后，对应到 ImageNet 的类别标签中可以找到，263 对应的是 Pembroke（柯基狗），这就证明模型已经加载成功了
'''

# 加载数据集
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       transform=transforms.ToTensor(),
                                       target_transform=None,
                                       download=True)
# 取32张图片的tensor
tensor_dataloader = DataLoader(dataset=cifar10_dataset,
                               batch_size=32)
data_iter = iter(tensor_dataloader)
img_tensor, label_tensor = data_iter.next()
print(img_tensor.shape)
grid_tensor = torchvision.utils.make_grid(img_tensor, nrow=16, padding=2)
grid_img = transforms.ToPILImage()(grid_tensor)

# grid_img.show()


# 提取分类层的输入参数
fc_in_features = alexnet.classifier[6].in_features
# 修改预训练模型的输出分类数
alexnet.classifier[6] = torch.nn.Linear(fc_in_features, 10)
# print(alexnet)

#     (6): Linear(in_features=4096, out_features=10, bias=True)  这时，你可以发现输出就变为 10 个单元了。

# 使用alexnet模型，CIFAR-10为数据集训练自己的模型


transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       transform=transform,
                                       target_transform=None,
                                       download=False)
dataloader = DataLoader(dataset=cifar10_dataset, # 传入的数据集, 必须参数
                               batch_size=32,       # 输出的batch大小
                               shuffle=True,       # 数据是否打乱
                               num_workers=2)      # 进程数, 0表示只有主进程

# 定义优化器
optimizer = torch.optim.SGD(alexnet.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

# 开始训练

# 训练3个Epoch
for epoch in range(3):
    for item in dataloader:
        output = alexnet(item[0])
        target = item[1]
        # 使用交叉熵损失函数
        loss = nn.CrossEntropyLoss()(output, target)
        print('Epoch {}, Loss {}'.format(epoch + 1 , loss))
        #以下代码的含义，我们在之前的文章中已经介绍过了
        alexnet.zero_grad()
        loss.backward()
        optimizer.step()
