from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds

def svhn_train_dataset(batch_size):
  
  train=tfds.load('svhn_cropped', split='train[:5%]', shuffle_files=True)
  train.download_and_prepare()
  train= tfds.as_numpy(train.as_dataset(split='train', batch_size=-1))
  X_train = train['image']
  y_train = train['label']

  return (
      tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
  )

def svhn_test_dataset():
    batch_size = 32

    train=tfds.builder('svhn_cropped')
    train.download_and_prepare()
    train= tfds.as_numpy(train.as_dataset(split='train', batch_size=-1))
    X_train = train['image']
    y_train = train['label']
    return (
      tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
  )

def build_and_compile_cnn_model():
  model = keras.Sequential()
  model.add(keras.Input(shape=(32, 32, 3)))  

  model.add(keras.layers.Conv2D(32, 3, activation="relu"))
  model.add(keras.layers.Conv2D(32, 3, activation="relu"))
  model.add(keras.layers.MaxPooling2D(2))
  model.add(keras.layers.Dropout(0.3))
  model.add(keras.layers.Conv2D(64, 3, activation="relu"))
  model.add(keras.layers.Conv2D(64, 3, activation="relu"))
  model.add(keras.layers.MaxPooling2D(2))
  model.add(keras.layers.Dropout(0.3))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(512, activation="relu"))
  model.add(keras.layers.Dropout(0.3))
  model.add(keras.layers.Dense(10, activation='softmax'))

  model.summary()

  model.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.SparseCategoricalCrossentropy(),
      metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )















  return model

