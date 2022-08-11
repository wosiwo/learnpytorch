
# 使用numpy处理图片
from PIL import Image
im = Image.open('images/1.jpeg')
print(im.size)

# 转为numpy

import numpy as np

im_pillow = np.asarray(im)

print(im_pillow.shape)