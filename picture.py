import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
import torch
from PIL import Image
import pandas as pd
import cv2
import scipy.io as scio
from torchvision.transforms import transforms

image_size = 64
dataFile = r'F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\datasets\freeform.mat'
data = scio.loadmat(dataFile)
print(data.keys())
pic = data.get('pattern').transpose(2, 0, 1)
parameter = data.get('parameter')
colMax = list(map(max, zip(*parameter)))  # 列最大值
pixel = parameter/colMax
print(colMax)
for i in range(0, 174883):
   layer1 = pic[i]*parameter[i][0]
   layer2 = pic[i]*parameter[i][1]
   layer3 = pic[i]*parameter[i][2]
   np.savez(r"F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\datasets\B\%s.npz" % i, layer1, layer2, layer3)
# dasta = np.load(r"F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\datasets\B\71120.npz")
# assd = np.stack([dasta.get('arr_0'), dasta.get('arr_1'), dasta.get('arr_2')], axis=0)

# final_pic = np.stack([layer1, layer2, layer3], axis=0)*255
# fig, ax = plt.subplots()
# plt.imshow(final_pic.transpose(1, 2, 0))
# #去白边
# plt.axis('off')
# height, width, channel = final_pic.transpose(1, 2, 0).shape
# fig.set_size_inches(width / 100.0, height / 100.0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.show()
# cv2.imwrite(r'F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\datasets\B\total\ii.bmp', final_pic.transpose(1, 2, 0))
# qp = cv2.imread(r'F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\datasets\B\total\ii.bmp').transpose(2, 0, 1)/255

