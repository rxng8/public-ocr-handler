import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers, Model

from utils import *

def linear_model(input_shape=(28,28)):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def conv_model(input_shape=(28, 28, 1)):
    in_tensor = layers.Input(shape=input_shape)
    # tensor = tf.expand_dims(in_tensor, axis=-1)
    tensor = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(in_tensor)
    tensor = layers.MaxPool2D()(tensor)
    tensor = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(tensor)
    tensor = layers.MaxPool2D()(tensor)
    tensor = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(tensor)
    tensor = layers.Flatten()(tensor)
    tensor = layers.Dense(32, activation='relu')(tensor)
    tensor = layers.Dense(16, activation='relu')(tensor)
    out_tensor = layers.Dense(10)(tensor)
    
    model = Model(inputs=in_tensor, outputs=out_tensor)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()
    return model

class OCRNet(Model):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.conv1 = make_down_conv_sequence(16)
        self.conv2 = make_down_conv_sequence(32)
        self.conv3 = make_down_conv_sequence(64)
        self.flatten = tf.keras.layers.Flatten()
        self.linear1 = make_dense_layer(512)
        self.linear2 = make_dense_layer(64)
        self.out = make_dense_layer(10, activation=None)

        self.model = tf.keras.Sequential([
            self.conv1,
            self.conv2,
            self.conv3,
            self.flatten,
            self.linear1,
            self.linear2,
            self.out
        ])

    def call(self, batch_input):
        return self.model(batch_input)

### Utilities methods

def make_dense_layer(out_channels, activation='relu'):
    return tf.keras.layers.Dense(out_channels, activation=activation)

def make_conv_layer(out_channels, strides=1, activation='relu', padding='same'):
    layer = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=(4, 4),
        strides=strides,
        activation=activation,
        padding=padding
    )
    return layer

def make_dropout_layer(rate=0.5):
    return tf.keras.layers.Dropout(rate)

def make_max_pooling_layer():
    return tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )

def make_batch_norm_layer(**kwargs):
    return tf.keras.layers.BatchNormalization(**kwargs)

def make_down_conv_sequence(out_channels, **kwargs):
    return tf.keras.Sequential([
        make_conv_layer(out_channels, **kwargs),
        make_max_pooling_layer(),
        make_dropout_layer()
    ])
