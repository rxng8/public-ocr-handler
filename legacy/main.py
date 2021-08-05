# %%

# import the necessary packages
from google.protobuf.text_format import PrintField
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import json

from tensorflow.keras.models import load_model
import tensorflow as tf

from utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

model = load_model("./models/ocrModel")

# %%

IMAGE_NAME = "credit_card_01.png"
IMAGE_PATH = os.path.join(EXAMPLE_DS, IMAGE_NAME)

# initialize a rectangular (wider than it is tall) and square
# structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
# sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(IMAGE_PATH)
image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Original, Shape:", image.shape, "Min:", np.min(image), "Max:", np.max(image))


# apply a tophat (whitehat) morphological operator to find light
# regions against a dark background (i.e., the credit card numbers)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

if verbose:
    print("tophat img, shape:", tophat.shape, "Min:", np.min(tophat), "Max:", np.max(tophat))
    show_img(tophat)

# compute the Scharr gradient of the tophat image, then scale
# the rest back into the range [0, 255]
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
    ksize=-1)

if verbose:
    print("gradX img, shape:", gradX.shape, "Min:", np.min(gradX), "Max:", np.max(gradX))
    show_img(gradX)

gradX = np.absolute(gradX)

if verbose:
    print("gradX img, shape:", gradX.shape, "Min:", np.min(gradX), "Max:", np.max(gradX))
    show_img(gradX)

(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

if verbose:
    print("gradX img, shape:", gradX.shape, "Min:", np.min(gradX), "Max:", np.max(gradX))
    show_img(gradX)

# apply a closing operation using the rectangular kernel to help
# cloes gaps in between credit card number digits, then apply
# Otsu's thresholding method to binarize the image
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

if verbose:
    print("Threshold img, shape:", thresh.shape, "Min:", np.min(thresh), "Max:", np.max(thresh))
    show_img(thresh)

# apply a second closing operation to the binary image, again
# to help close gaps between credit card number regions
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

print("Threshold img, shape:", thresh.shape, "Min:", np.min(thresh), "Max:", np.max(thresh))
show_img(thresh)

# find contours in the thresholded image, then initialize the
# list of digit locations
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
locs = []

# %%

# loop over the contours
for (i, c) in enumerate(cnts):
    # compute the bounding box of the contour, then use the
    # bounding box coordinates to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    
    # tmp = cv2.rectangle(gray.copy(), (x, x+w), (y, y+ h), (0,255,0), 1)
    # show_img(tmp)

    ar = w / float(h)
    # since credit cards used a fixed size fonts with 4 groups
    # of 4 digits, we can prune potential contours based on the
    # aspect ratio
    if ar > 2.5 and ar < 4.0:
        # contours can further be pruned on minimum/maximum width
        # and height
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # append the bounding box region of the digits group
            # to our locations list
            locs.append((x, y, w, h))

# sort the digit locations from left-to-right, then initialize the
# list of classified digits
locs = sorted(locs, key=lambda x:x[0])
output = []

# loop over the 4 groupings of 4 digits
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # initialize the list of group digits
    groupOutput = []

    # extract the group ROI of 4 digits from the grayscale image,
    # then apply thresholding to segment the digits from the
    # background of the credit card
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # detect the contours of each individual digit in the group,
    # then sort the digit contours from left to right
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
    digitCnts = imutils.grab_contours(digitCnts)
    digitCnts = contours.sort_contours(digitCnts,
        method="left-to-right")[0]

    # loop over the digit contours
    for c in digitCnts:
        # compute the bounding box of the individual digit, extract
        # the digit, and resize it to have the same fixed size as
        # the reference OCR-A images
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]

        # Roi range from 0 to 255
        # preprocess roi
        roi = preprocess(roi)
        
        show_img(roi)
        # print(roi.shape)

        # roi is the input
        batch_roi = tf.expand_dims(roi, axis=0)
        result = model.predict(batch_roi)[0]

        # Apeend the number to the group output !!!
        groupOutput.append(str(np.argmax(result)))

    # draw the digit classifications around the group
    cv2.rectangle(image, (gX - 5, gY - 5),
        (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # update the output digits list
    output.extend(groupOutput)

# display the output credit card information to the screen
# print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
# cv2.imshow("Image", image)
# show_img(image)




# %%
