from PIL import Image
from torchvision import transforms

img = Image.open('./images/jk.jpg')
img.show()
# display(img)
print(type(img))  # PIL.Image.Image是PIL.JpegImagePlugin.JpegImageFile的基类
'''
输出: 
<class 'PIL.JpegImagePlugin.JpegImageFile'>
'''

# PIL.Image转换为Tensor
img1 = transforms.ToTensor()(img)
print(type(img1))
'''
输出: 
<class 'torch.Tensor'>
'''

# Tensor转换为PIL.Image
img2 = transforms.ToPILImage()(img1)  # PIL.Image.Image
print(type(img2))
img2.show()
'''
输出: 
<class 'PIL.Image.Image'>
'''
