import keras
from keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, Dropout, Activation, Flatten
import numpy as np

model = Sequential()

# Block 1
model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same', input_shape=(224, 224, 3)))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Block 2
model.add(Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Block 3
model.add(Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(Conv2D(filters=265, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(Conv2D(filters=265, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Block 4
model.add(Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Block 5
model.add(Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Block 6
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

print(model.summary(), file=open("vggnet_summary.txt", "a"))