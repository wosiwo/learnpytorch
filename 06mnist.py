# 以MNIST为例
import torchvision

mnist_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=None,
                                           target_transform=None,
                                           download=True)
mnist_dataset_list = list(mnist_dataset)
print(mnist_dataset_list)