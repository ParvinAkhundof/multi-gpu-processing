import tensorflow as tf
import numpy as np
from tensorflow import keras

def mnist_dataset_train(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the [0, 255] range.
  # You need to convert them to float32 with values in the [0, 1] range.
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).batch(batch_size)
  return train_dataset

def mnist_dataset_test(batch_size):
  _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the [0, 255] range.
  # You need to convert them to float32 with values in the [0, 1] range.
  x_test = x_test / np.float32(255)
  y_test = y_test.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_test, y_test)).batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = keras.Sequential()
  model.add(keras.Input(shape=(28, 28, 1)))  
  # model.add(keras.layers.Conv2D(32, 3, activation="relu"))
  # model.add(keras.layers.Flatten())
  # model.add(keras.layers.Dense(128, activation="relu"))
  # model.add(keras.layers.Dense(10, activation='softmax'))
  model.add(keras.layers.Conv2D(32, 3, activation="relu"))
  model.add(keras.layers.Conv2D(32, 3, activation="relu"))
  model.add(keras.layers.MaxPooling2D(2))
  model.add(keras.layers.Dropout(0.3))
  model.add(keras.layers.Conv2D(64, 3, activation="relu"))
  model.add(keras.layers.Conv2D(64, 3, activation="relu"))
  model.add(keras.layers.MaxPooling2D(2))
  model.add(keras.layers.Dropout(0.3))

  model.add(keras.layers.Conv2D(128, 3, activation="relu"))
  model.add(keras.layers.Conv2D(128, 3, activation="relu"))
  model.add(keras.layers.MaxPooling2D(2))
  model.add(keras.layers.Dropout(0.3))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(512, activation="relu"))
  model.add(keras.layers.Dropout(0.3))
  model.add(keras.layers.Dense(10, activation='softmax'))
  

  model.summary()
  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model

