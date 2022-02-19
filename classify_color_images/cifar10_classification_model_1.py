### IMPORT PACKAGES ###
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from numba import cuda

### LOAD DATA ###
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Plot some figures to see how color images look
fig = plt.figure(figsize = (20,5))
for i in range(36):
    ax = fig.add_subplot(3,12, i+1, xticks=[], yticks=[])
    ax.imshow((x_train[i]))

# plt.show()
plt.savefig("Image samples from CIFAR10", dpi=300)

# Store y_train label values
label_values = {0:'airplane', 1:'automobile', 2:'bird',
               3:'cat', 4:'deer', 5:'dog', 6:'frog',
               7:'horse', 8:'ship', 9:'truck'}

### PREPROCESS ###
# Normalize images sizes
x_train = np.float16(x_train) / 255 # Divide by 255 to get range 0 to 1
x_test = np.float16(x_test) / 255

# OneHotEncode Labels
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train).toarray()
y_test = ohe.fit_transform(y_test).toarray()

# Separate training data into training and validation, leave test data alone
x_train, x_valid = x_train[5000:], x_train[:5000]
y_train, y_valid = y_train[5000:], y_train[:5000]

print('Training samples: ', x_train.shape[0])
print('Validation Samples: ', x_valid.shape[0])
print('Test Samples: ', x_test.shape[0])

### Neural Network Architecture ###
# Instantiate Model
model = Sequential()

# First layer --- NEED TO INCLUDE INPUT SHAPE (32,32,3)
model.add(Conv2D(filters=16, kernel_size=2, padding='same',
                 activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))

# Second layer
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# Third Layer
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# First dropout layer to prevent over fitting
model.add(Dropout(0.3))

# Flatten image to feed to dense layers
model.add(Flatten())

# First fully connected layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))

# Output Dense layer - USE SOFTMAX with 10 outputs
model.add(Dense(10, activation='softmax'))

model.summary()

### COMPILE MODEL ###
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpoints = ModelCheckpoint(filepath='cifar10_model_1.keras', verbose=1, save_best_only=True)

training = model.fit(x_train, y_train, batch_size=32, epochs=5,
                     validation_data=(x_valid,y_valid), callbacks=[checkpoints],
                     verbose=1, shuffle=True)

### LOAD BEST MODEL ###
model = load_model('cifar10_model_1.keras')
score = model.evaluate(x_test, y_test, verbose=2)
print('\n', 'Test accuracy: ', score[1])