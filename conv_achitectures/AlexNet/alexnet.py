import keras
import tensorflow as tf
from keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, Dropout, Activation, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np

# Load data: CIFAR 10 will overfit, not complex enough data for AlexNet
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(x_test.shape)

### PREPROCESS DATA ###
# Normalize x_train and x_test
mean = np.mean(x_train)
std = np.std(x_test)
x_train = (x_train-mean)/(std + 1e-7)
x_test = (x_test-mean)/(std + 1e-7)

# OneHotEncode labels
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

strategy = tf.distribute.MirroredStrategy()
print("Number of GPU's: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    # AlexNet Architecture (Adapted for smaller image size)
    # CONV -> Pool -> CONV -> Pool -> CONV -> CONV -> CONV -> POOL -> FLATTEN -> FC -> FC
    model = Sequential()

    # Conv1 with ReLu Activation and maxpooling
    model.add(Conv2D(filters=96, kernel_size=3, strides=2, input_shape=(32, 32, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(BatchNormalization())

    # 2nd layer (conv + pool + batchnorm)
    model.add(Conv2D(filters=256, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(0.005)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    #
    # Conv3 with NO POOL, and BatchNorm
    model.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # Conv4 with NO POOL, and BatchNorm
    model.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # Conv5 with POOL, and BatchNorm
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.005)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=3, strides=2))

    # Flatten
    model.add(Flatten())

    # FC1 + Dropout to stop overfitting (Dropout needed normally for Fully Connected)
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.8))

    # FC2 + Dropout to stop overfitting (Dropout needed normally for Fully Connected)
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.6))

    # Output layer
    model.add(Dense(10, activation='softmax'))

    model.summary()

    reduce_learning = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1))
    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=90, validation_data=(x_test, y_test),
          verbose=2, callbacks=[reduce_learning])

