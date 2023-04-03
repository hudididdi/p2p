import cv2
from pyts.image import MarkovTransitionField
import numpy as np
import os
import scipy.io as scio

for i in range(0, 175000):
    if os.path.exists(
            r"F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\to\data\A\test\%s.npz" % i):
        A = np.load(r"F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\to\data\A\test\%s.npz" % i)
        B = np.load(r"F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\to\data\B\test\%s.npz" % i)
        layer1A = A.get('arr_0')
        layer2A = A.get('arr_1')
        layer3A = np.zeros((64, 64))
        layer1B = B.get('arr_0') / 3
        layer2B = B.get('arr_1') / 1
        layer3B = B.get('arr_2') / 5
        picA = np.stack([layer1A[0], layer2A[0], layer3A], axis=0).transpose(1, 2, 0)
        picB = np.stack([layer1B, layer2B, layer3B], axis=0).transpose(1, 2, 0)
        im_AB = np.concatenate([picA, picB], 1).astype(np.float32)
        cv2.imwrite(r"F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\to1\data\A\test\%s.tif" % i, im_AB)

# debugimg = im_AB.transpose(2, 0, 1)
# ass = cv2.imread(r'C:\Users\pxy\Desktop\1.jpg')
# ab = cv2.imread(
#     r'F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\to1\data\A\test\1.tif').transpose(2, 0, 1)