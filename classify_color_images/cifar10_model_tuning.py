### IMPORT PACKAGES ###
# Neural Framework Packages
import keras
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import regularizers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


# Packages for Datasets
from keras.datasets import cifar10

# Other Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import time

runtime_name = "cifar10_conv6_dense2_{}".format(int(time.time()))

# Tensorboard setup
tensorboard =TensorBoard(log_dir='logs/{}'.format(runtime_name),  update_freq='epoch', profile_batch=0)
### LOAD DATA ###
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.float32(x_train)
x_test = np.float32(x_test)

# Split training into train and validation
x_train, x_val = x_train[5000:], x_train[:5000]
y_train, y_val = y_train[5000:], y_train[:5000]

# print(x_train.shape) -> (45000, 32, 32, 3)
# print(x_val.shape) -> (5000, 32, 32, 3)
# print(x_test.shape) -> (10000, 32, 32, 3)

# Normalize data
mean = np.mean(x_train)
stdev = np.mean(x_train)
x_train = (x_train-mean)/stdev
x_val = (x_val-mean)/stdev
x_test = (x_test-mean)/stdev

# OneHotEncode y labels
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_val = ohe.transform(y_val)
y_test = ohe.transform(y_test)


# Image Generator
aug = ImageDataGenerator(
    rotation_range = 30,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False
)
aug.fit(x_train)


### NETWORK ARCHITECTURE ###
hidden_units = 32
weight_decay = 1e-4 # L2 Regularization parameter

model = Sequential()

# LAYER 1: CONV #
model.add(Conv2D(hidden_units, kernel_size=3,
                 padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())

# LAYER 2: CONV #
model.add(Conv2D(hidden_units, kernel_size=3,
                 padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# LAYER 3: POOLING/DROPOUT #
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# LAYER 4: CONV #
model.add(Conv2D(hidden_units * 2, kernel_size=4,
                 padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# LAYER 5: CONV #
model.add(Conv2D(hidden_units * 2, kernel_size=4,
                 padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# LAYER 6: POOLING/DROPOUT #
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

# LAYER 7: CONV #
model.add(Conv2D(hidden_units * 4, kernel_size=3,
                 padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# LAYER 8: CONV #
model.add(Conv2D(hidden_units * 4, kernel_size=3,
                 padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# LAYER 9: POOLING/DROPOUT #
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# LAYER 10: CONV #
model.add(Conv2D(hidden_units * 8, kernel_size=2,
                 padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# LAYER 11: CONV #
model.add(Conv2D(hidden_units * 8, kernel_size=2,
                 padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.45))

# LAYER 12: DENSE #
model.add(Flatten())
model.add(Dense(hidden_units * 16, activation='relu'))
model.add(Dropout(0.5))

# LAYER 14: DENSE #
model.add(Dense(hidden_units * 16, activation='relu'))
model.add(Dropout(0.5))

# LAYER 15: DENSE #
model.add(Dense(10, activation='softmax', dtype='float32'))

model.summary()

### MODEL TRAINING ###
batch_size = 256
epochs = 200

checkpoints = ModelCheckpoint(filepath='cifar10_model_tuned.keras', verbose=1, save_best_only=True)
csv_logger = CSVLogger('training_hist.csv', separator=',', append=False)

optimizer = optimizers.Adam(lr=0.0005)
# Convert to mixed precision

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

fitted = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs,verbose=1, callbacks=[checkpoints, tensorboard, csv_logger],
                validation_data=(x_val,y_val))

scores = model.evaluate(x_test, y_test, batch_size = batch_size, verbose=1)
print(f"Test Results: {scores[1]*100}, Loss: {scores[0]}")

plt.plot(fitted.history['accuracy'], label='train')
plt.plot(fitted.history['val_accuracy'], label='test')
plt.title("Tuned Model CIFAR10")
plt.legend()
plt.savefig('cifar10_tuned.png', dip=300)






