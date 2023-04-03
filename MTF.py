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
real = data.get('real')
imag = data.get('imag')
Transf = np.empty((3, 64, 64))
Third = np.zeros((64, 64))
fs = np.empty((1, len(real[0])), dtype=complex)

for i in range(0, 174883):
    fs = np.empty((1, len(real[0])), dtype=complex)
    for j in range(0, 301):
       fs[0][j] = complex(real[i][j], imag[i][j])
    fz = np.abs(fs)
    xw = np.angle(fs, deg=True)
    mtf = MarkovTransitionField(image_size=image_size)
    mtf_fz = mtf.fit_transform(fz)
    mtf_xw = mtf.fit_transform(xw)
    Transf = np.stack([mtf_fz[0], mtf_xw[0]], axis= 0)
    np.savez(r"F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\datasets\A\%s.npz" % i, mtf_fz, mtf_xw)
# dasta = np.load(r"F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\datasets\A\total\II.npz")
# asas = dasta.get('arr_0')
# aas = dasta.get('arr_1')

# fig, ax = plt.subplots(1, 1, constrained_layout=True)
# ax.imshow(mtf_fz[0])
# ax.imshow(mtf_xw[0])
# ax.imshow(Transf)
# plt.show()

# x = np.arange(30, 60.1, 0.1)
# fig, ax = plt.subplots(2, 1)
#
# ax[0].plot(x, fz[0], ls='-', lw=1.0, label="CNN_Train")
# ax[0].set_xticks(np.arange(30, 61, 2.5))
# ax[0].set_yticks(np.arange(0, 1.1, 0.2))
# ax[0].set_xlim((30, 60))
# ax[0].set_ylim((0, 1))
#
# ax[1].plot(x, xw[0], ls='-', lw=1.0, label="CNN_ain")
# ax[1].set_xticks(np.arange(30, 61, 2.5))
# ax[1].set_yticks(np.arange(-180, 181, 30))
# ax[1].set_xlim((30, 60))
# ax[1].set_ylim((-200, 200))
# plt.show()

# image.imsave(r"F:\resnet\zhuanhua/MTF.png", sin_gasf[0])  # 保存图片 (save image)
# np.savetxt(r"F:\resnet\zhuanhua/MTF.csv", sin_gasf[0])  # 保存数据为 csv 文件
# imagess = cv2.imread(r"F:\resnet\zhuanhua/MTF.png")
# a = np.loadtxt(r"F:\resnet\zhuanhua/MTF.csv")

# fig, ax = plt.subplots(1, 1, constrained_layout=True)
# ax.imshow(a)
# plt.show()
# a