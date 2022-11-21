
from scipy.io import loadmat

train = loadmat('train_32x32.mat')
test = loadmat('test_32x32.mat')

# train and test are python dictionaries
# keys are ['__header__', '__version__', '__globals__', 'X', 'y']

X_train = train['X']
y_train = train['y']
X_test = test['X']
y_test = test['y']

# X and y are numpy ndarrays

print("Shape of X_train is:", X_train.shape)
print("Shape of y_train is:", y_train.shape)
print("Shape of X_test is:", X_test.shape)
print("Shape of y_test is:", y_test.shape)


import matplotlib.pyplot as plt


X_train = X_train[:,:,:,:]
X_test = X_test[:,:,:,:]

start = 50

fig, ax = plt.subplots(3, 4)
for row in range(3):
    for col in range(4):
        ax[row, col].imshow(X_train[:,:,:,start+row*3+col])
        ax[row, col].axis('off')
        ax[row, col].set_title('{}'.format(y_train[start+row*3+col, 0]))
# plt.show()

import numpy as np

X_train = np.rollaxis(X_train, 3)
X_test = np.rollaxis(X_test, 3)

print("Shape of X_train is now:", X_train.shape)
print("Shape of X_test is now:", X_test.shape)

y_train = y_train[:,0]
y_test = y_test[:,0]

print("Shape of y_train is now:", y_train.shape)
print("Shape of y_test is now:", y_test.shape)

y_train[y_train==10] = 0
y_test[y_test==10] = 0

print("labels of y_train are", np.unique(y_train[:]))
print("labels of y_test are", np.unique(y_test[:]))

X_train = X_train / 255
X_test = X_test / 255

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Conv2D(64, (3,3), activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# get a summary of our built model
model.summary()

# define the optimizer, loss function and metrics for the network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# start training
model.fit(X_train, y_train, epochs=2)  #10

loss, acc = model.evaluate(X_test, y_test)
print("Model accuracy on test data is: {:6.3f}%".format(100 * acc))

model.save('my_model.h5')