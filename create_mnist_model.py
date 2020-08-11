import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import numpy as np


# load the MNIST data set and split it into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data into proper format (4D tensor)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# convert the data to the proper type
x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices (helps calculate crossentropy loss later on)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#create model
model = Sequential()

# convolve image with 32 5x5 filters followed by 2x2 pooling
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape= (28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#convolve pooled layer 1 feature map with 64 5x5 layers followed by more pooling
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#flatten model, go from 7x7x64 nodes to 1000, then arrive to 10 nodes in output later (0-9)
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

#compile model with Adam optimizer and Keras built-in categorical cross entropy tool
model.compile(loss= tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

#class to keep track of callbacks
class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

#train model with 15 epochs and all of the other data specified
model.fit(x_train, y_train,
          batch_size = 128,
          epochs = 5,
          verbose = 1,
          validation_data = (x_test, y_test),
          callbacks = [history])

model.save('mnist_cnn.h5')
print("Model saved")
