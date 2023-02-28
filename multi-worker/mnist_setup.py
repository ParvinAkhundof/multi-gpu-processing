import tensorflow as tf
import numpy as np
from tensorflow import keras

def mnist_dataset_train(batch_size,index,num_workers):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.float32)


  # y_train = tf.keras.utils.to_categorical(y_train, 10)




  # x_train=np.concatenate((x_train, x_train), axis=0)
  # y_train=np.concatenate((y_train, y_train), axis=0)

  # x_train=np.concatenate((x_train, x_train), axis=0)
  # y_train=np.concatenate((y_train, y_train), axis=0)
  
  # x_train=np.concatenate((x_train, x_train), axis=0)
  # y_train=np.concatenate((y_train, y_train), axis=0)
  
  # x_train=np.concatenate((x_train, x_train), axis=0)
  # y_train=np.concatenate((y_train, y_train), axis=0)
  
  # x_train=np.concatenate((x_train, x_train), axis=0)
  # y_train=np.concatenate((y_train, y_train), axis=0)


  # x_train=np.array_split(x_train, num_workers)[index]
  # y_train=np.array_split(y_train, num_workers)[index]

  print(num_workers)
  print(x_train.size)

  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).batch(batch_size)
  return train_dataset

def mnist_dataset_test(batch_size,index,num_workers):
  _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_test = x_test / np.float32(255)
  y_test = y_test.astype(np.float32)

  # y_test = tf.keras.utils.to_categorical(y_test, 10)
  
  # x_test=np.array_split(x_test, num_workers)[index]
  # y_test=np.array_split(y_test, num_workers)[index]

  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_test, y_test)).batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  # model = keras.Sequential()
  # model.add(keras.Input(shape=(28, 28, 1))) 
  # model.add(keras.layers.Conv2D(32, (3,3), activation="relu"))
  # model.add(keras.layers.Conv2D(32, (3,3), activation="relu"))
  # model.add(keras.layers.MaxPooling2D(2,2))
  # model.add(keras.layers.Dropout(0.3))
  # model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
  # model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
  # model.add(keras.layers.MaxPooling2D(2,2))
  # model.add(keras.layers.Dropout(0.3))
  # model.add(keras.layers.Flatten())

  # model.add(keras.layers.Dense(512, activation="relu"))
  # model.add(keras.layers.Dropout(0.3))
  # model.add(keras.layers.Dense(10, activation='softmax'))


  input_shape = (28, 28, 1)
  num_classes = 10
  model = tf.keras.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    # Second convolutional layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    # Third convolutional layer
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    # Flatten the output from the convolutional layers
    tf.keras.layers.Flatten(),
    # Fully connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    # Output layer with softmax activation
    tf.keras.layers.Dense(num_classes, activation='softmax')
  ])
  


  model.summary()
  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

