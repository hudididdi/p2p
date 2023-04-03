import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool

path_A = r'F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\to\data\A\train\1.npz'
path_B = r'F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\to\data\B'
path_AB = r'F:\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\to\data\im_AB'
im_A = cv2.imread(path_A)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
im_B = cv2.imread(path_B, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
# im_AB = np.concatenate([im_A, im_B], 1)
# cv2.imwrite(path_AB, im_AB)
a

