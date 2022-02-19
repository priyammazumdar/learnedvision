### LeNet Model Architecture:
### C1 = 6 Layers, c2 = 16 layers, c3 = 120 layers
### Kernel Size of Conv Layers is 5x5
### Pooling after each Conv: 2x2 receptor with AvgPooling
### Tanh activation function with
### READ SECTION 2 in lecun-98 for more details ###

### Implementation of LeNet Model ###
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, AveragePooling2D, Activation
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import time

# Load MNIST DATASET
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data
mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print(f"X_train Shape: {X_train.shape}")

# OneHotEncode Labels
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

print(f"X_test Shape: {y_train.shape}")

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Set mixed precision throughout
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

model = Sequential()

# Conv1 Layer (6 Filters) with tanh activation
model.add(Conv2D(filters=6, kernel_size=5, strides=1, padding='same', input_shape=(28, 28, 1)))
model.add(Activation('tanh'))

# Pool1 Layer with AveragePooling
model.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))

# Conv2 Layer (16 filters) with tanh activation
model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='valid'))
model.add(Activation('tanh'))

# Pool2 Layer with Average Pooling
model.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))

# Conv3 Layer with 120 filters
model.add(Conv2D(filters=120, kernel_size=5, strides=1, padding='valid'))
model.add(Activation('tanh'))

# Flatten Conv Layers
model.add(Flatten())

# FC1 with 84 units with tanh activation
model.add(Dense(units=84))
model.add(Activation('tanh'))

# FC2 with 10 outputs units and Softmax activation
model.add(Dense(units=10, dtype='float32'))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Stepwise Learning Rate Decay used in LeNet-5
def lr_schedule(epoch):
    if epoch <= 2:
        lr = .0005
    elif (epoch > 2) and (epoch <= 5):
        lr = .0002
    elif (epoch > 5) and (epoch <= 9):
        lr = .00005
    else:
        lr = .00001

    return lr


# Tensorboard Run Name
runtime_name = "LeNet5_{}".format(int(time.time()))

# Callbacks
lr_scheduler = LearningRateScheduler(lr_schedule)
tensorboard = TensorBoard(log_dir='logs/{}'.format(runtime_name), update_freq='epoch', profile_batch=0)
checkpoints = ModelCheckpoint(filepath='lenet5_mnist.keras', verbose=1, save_best_only=True)
csv_logger = CSVLogger('training_hist.csv', separator=',', append=False)

callbacks = [lr_scheduler, tensorboard, checkpoints, csv_logger]

model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test), callbacks=callbacks,
          verbose=1, shuffle=True)
