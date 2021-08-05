# %%

# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model
import tensorflow as tf

from utils import *
from models import OCRNet
from data import get_train_generator

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMG_SIZE = (28, 28)
BATCH_SIZE = 32

DATASET_PATH = "./dataset/separate/Credit Card Number Dataset"

train_gen = get_train_generator(DATASET_PATH, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

model = OCRNet(batch_size=BATCH_SIZE)

# %%

model.compile(
	optimizer=tf.keras.optimizers.Adam(0.001),
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.build(input_shape=(BATCH_SIZE, *IMG_SIZE, 1))
model.summary()

# %%

EPOCHS = 30
STEP_PER_EPOCHS = 20

model.fit(
	train_gen,
	steps_per_epoch=STEP_PER_EPOCHS,
	epochs=EPOCHS
)

# %%

MODEL_PATH = "./models/ocrModel"
model.save(MODEL_PATH)


# %%

####### Advanced Training ######

# # Train definition

# optimizer = optimizers.Adam()

# def loss_function(real, pred):
# 	return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(real, pred)

# @tf.function
# def train_step():
# 	pass


# ### Training ###

# checkpoint_path = None

# EPOCHS = 1
# STEP_PER_EPOCHS = 20

# with tf.device('/device:CPU:0'):
#     for epoch in range(EPOCHS):
#         print("\nStart of epoch %d" % (epoch + 1,))

#         # Iterate over the batches of the dataset.
#         for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
#             logits_human, logits_mask = train_step(x_batch_train, y_batch_train)
            
#         # For each epoch, save checkpoint
#         # model.save_weights(checkpoint_path)
#         # print("Checkpoint saved!")


