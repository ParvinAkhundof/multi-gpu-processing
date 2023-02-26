from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds

def svhn_train_dataset(batch_size,index,num_workers):
  train = loadmat('../train_32x32.mat')
  X_train = train['X']
  y_train = train['y']
  X_train = np.rollaxis(X_train, 3)/ 255
  y_train = y_train[:,0]
  y_train[y_train==10] = 0

  X_train=np.array_split(X_train, num_workers)[index]
  y_train=np.array_split(y_train, num_workers)[index]

  return (
      tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
  )

def svhn_test_dataset(batch_size,index,num_workers):

    test = loadmat('../test_32x32.mat')

    X_test = test['X']
    y_test = test['y']
    X_test = np.rollaxis(X_test, 3)/ 255
    y_test = y_test[:,0]
    y_test[y_test==10] = 0

    # X_test=np.array_split(X_test, num_workers)[index]
    # y_test=np.array_split(y_test, num_workers)[index]

    return (
        tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    )
  
  


def build_and_compile_cnn_model():
  model = keras.Sequential()
  model.add(keras.Input(shape=(32, 32, 3)))  

  model.add(keras.layers.Conv2D(2700, 3, activation="relu"))
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

