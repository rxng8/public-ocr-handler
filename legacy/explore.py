# %%

# import the necessary packages
from re import S
from imutils import contours
import numpy as np
import imutils
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import json

from utils import *

EXAMPLE_DS = "./images"

verbose = False

def apply_canny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)


# define a dictionary that maps the first digit of a credit card
# number to the credit card type
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# %%

IMAGE_NAME = "credit_card_02.png"
IMAGE_PATH = os.path.join(EXAMPLE_DS, IMAGE_NAME)
img = cv2.imread(IMAGE_PATH)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_img(img)

# %%

# Thresholding

# applying different thresholding
# techniques on the input image
# all pixels value above 120 will
# be set to 255
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
show_img(thresh1)
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
show_img(thresh2)
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
show_img(thresh3)
ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
show_img(thresh4)
ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
show_img(thresh5)


# %%

## Contour
# contours is the list of contuor in the image, 
# each contuor is a 3d numpy array that list all the pixel
# in the image.
# heirarchy is the list of relationship among contours

#get threshold image
ret, thresh_img = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

#find contours
_contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
img_contours = np.zeros(img.shape)
# draw the contours on the empty image
# To draw all the contours in an image:
cv2.drawContours(img_contours, _contours, -1, (0,255,0), 3)
show_img(img_contours)

# To draw an individual contour, say 4th contour:
img_contours = np.zeros(img.shape)
cv2.drawContours(img_contours, _contours, 3, (0,255,0), 3)
show_img(img_contours)

# But most of the time, below method will be useful
img_contours = np.zeros(img.shape)
cnt = _contours[3]
cv2.drawContours(img_contours, [cnt], 0, (0,255,0), 3)
show_img(img_contours)

# %%

## Morphological transformation

kernel = np.ones((5,5),np.uint8)
# If use other structuring element shape
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
erosion = cv2.erode(img_gray, kernel, iterations=1)
show_img(erosion)
dilation = cv2.dilate(img_gray, kernel, iterations=1)
show_img(dilation)
opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
show_img(opening)
closing = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
show_img(closing)
gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
show_img(gradient)
tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
show_img(tophat)
blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
show_img(blackhat)

