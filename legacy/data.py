# %%

# import the necessary packages
import imutils
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import *

def preprocessing_function(img):
    """function that will be applied on each input. The function 
    will run after the image is resized and augmented. The function 
    should take one argument: one image (Numpy tensor with rank 3 or 1,
    in this case, 1), and should output a Numpy tensor with the same 
    shape.

    TODO: Have some way of data generator that can invert the data so
    that the data will have more variances. It is also because that there
    are some cases in reality that the image of credit cards is contasted,
    and therefore causing the background to be white, and the actual number
    is black.

    Args:
        img (Tensor): Have shape (h, w, 1).

    Returns:
        Tensor: Preprocessed image.
    """
    assert len(img.shape) == 3, "Wrong shape"
    assert img.shape[2] == 1, "Wrong channels "
    rescaled = img.copy()
    if tf.reduce_max(img) > 1:
        # Cast image to range
        rescaled = tf.cast(img, tf.float32) / 255.

    # resize
    resized = tf.image.resize(
        rescaled, 
        (28, 28), 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return resized

def get_train_generator(DATASET_PATH, img_size=(28,28), batch_size=32, shuffle_buffer_size=1000):
    """Get the tensorflow training data gen

    TODO: Use the most up-to-date tf class Dataset taht consume data from
    generator, and shuffle and repeat the dataset! This has some conflict
    tf.image.ImageDatasetGenerator.flow_from_directory()
    

    Args:
        DATASET_PATH ([type]): [description]
        img_size (tuple, optional): [description]. Defaults to (28,28).
        batch_size (int, optional): [description]. Defaults to 32.
        shuffle_buffer_size (int, optional): [description]. Defaults to 1000.

    Returns:
        [type]: [description]
    """
    
    train_datagen = ImageDataGenerator(
        data_format='channels_last',
        preprocessing_function=preprocessing_function
    )

    # Deprecated
    # ds = tf.data.Dataset.from_generator(
    #     lambda: train_datagen.flow_from_directory(
    #         DATASET_PATH,
    #         target_size=img_size,
    #         batch_size=batch_size,
    #         class_mode='sparse',
    #         classes=['0','1','2','3','4','5','6','7','8','9'],
    #         color_mode='grayscale'
    #     ),
    #     output_signature=(
    #         tf.TensorSpec(shape=(batch_size, *img_size, 1), dtype=tf.float32),
    #         tf.TensorSpec(shape=(batch_size), dtype=tf.float32),
    #     )
    # )

    # return ds.repeat()

    return train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        classes=['0','1','2','3','4','5','6','7','8','9'],
        color_mode='grayscale'
    )


def examine_data(DATASET_PATH="./dataset/separate/Credit Card Number Dataset"):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    ##### Example data

    IMAGE_NAME = "5_106.png"
    IMAGE_PATH = os.path.join(DATASET_PATH, "5", IMAGE_NAME)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(IMAGE_PATH)
    image = imutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    show_img(gray)

    # remove noise
    gray = cv2.GaussianBlur(gray,(3,3),0)

    show_img(gray)

    grey = rgb2gray(image)

    show_img(grey)
    print(grey.shape)
    #### Examine wavelet of the data

    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)  # y

    # show_img(sobelx)

    plt.subplot(2,2,1),plt.imshow(gray,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()


def test(DATASET_PATH = "./dataset/separate/Credit Card Number Dataset"):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_datagen = ImageDataGenerator(
        data_format='channels_last',
        preprocessing_function=preprocessing_function
    )

    train_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=(28, 28),
            batch_size=32,
            class_mode='sparse',
            classes=['0','1','2','3','4','5','6','7','8','9'],
            color_mode='grayscale'
        )

    print(train_generator.class_indices)


    ds = tf.data.Dataset.from_generator(
        lambda: train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=(28, 28),
            batch_size=32,
            class_mode='sparse',
            classes=['0','1','2','3','4','5','6','7','8','9'],
            color_mode='grayscale'
        ), 
        output_signature=(
            tf.TensorSpec(shape=(32, 28, 28, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(32), dtype=tf.float32),
        )
    )

    ds.element_spec


    img, label = next(train_generator)


    show_img(img[5])
    print(img[5].shape)
    print(label[5])



