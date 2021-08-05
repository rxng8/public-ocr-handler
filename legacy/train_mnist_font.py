# %%

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers, Model

from tensorflow.keras.models import load_model

from utils import *
from models import linear_model, conv_model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# %%

# Code to test and see the shape of the dataset
import random
for batch_img, batch_label in ds_test:
    r = random.randint(0, batch_img.shape[0] - 1)
    show_img(batch_img[r])
    print(batch_label[r])
    break

# %%

# Code to get the convolutional model
model = conv_model()

# %%

# Code to fit model
model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
)

# %%

# Code to save model
model.save("./models/mnist_conv")


# %%

## Code to load model
model = load_model("./models/ocrModel")

