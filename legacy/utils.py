
import imutils
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import numpy as np

import tensorflow as tf

def show_img(img):
    if len(img.shape) == 3:
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    elif len(img.shape) == 2:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()


def preprocess(img):
    # Expect img to have 1 channel, range from [0, 255]
    
    # Image resize
    prep = cv2.resize(img, (20, 20))

    # Image expand border
    prep = np.asarray(prep)
    prep = Image.fromarray(prep)
    prep = ImageOps.expand(prep,border=4,fill='black')
    prep = np.asarray(prep)

    # Cast image to range
    prep = tf.cast(prep, tf.float32) / 255.

    # Convert to 1 channel image
    if len(prep.shape) == 2:
        prep = tf.expand_dims(prep, axis=-1)

    return prep


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray