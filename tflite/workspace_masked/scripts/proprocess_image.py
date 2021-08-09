# %%

from numpy import random
import pandas as pd
import numpy as np
import collections
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
import os

TEST_PATH = "../images/7_0_jpg.rf.290881ee5a45ee3ab54a5c54f68b4d40.jpg"

IMAGE_FOLDER = "../images/"

# %%

img = cv2.imread(TEST_PATH, 0)

# %%

# TEST CANNY EDGE DETECTION METHOD

edges = cv2.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# %%

# TEST CANNY EDGE DETECTION METHOD for all files

for fname in os.listdir(IMAGE_FOLDER):
    try:
        img = cv2.imread(os.path.join(IMAGE_FOLDER, fname), 0)
        edges = cv2.Canny(img,100,200)
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
    except:
        print(f"Error opening file {fname}")
