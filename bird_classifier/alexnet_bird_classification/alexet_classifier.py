### IMPORT DATA ###
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend
from keras.layers import Flatten, Dense, Conv2D, Activation, MaxPooling2D, BatchNormalization, Dropout
from keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.regularizers import l2
import time

backend.clear_session()

batch_size = 256
# Pull data and normalize image values
train_datagen = ImageDataGenerator(rescale=1/255,
                                   horizontal_flip=True,
                                   vertical_flip=False
                                   )
test_datagen = ImageDataGenerator(rescale=1/255,
                                  horizontal_flip=True,
                                  vertical_flip=False
                                  )
val_datagen = ImageDataGenerator(rescale=1/255,
                                 horizontal_flip=True,
                                 vertical_flip=False
                                 )

# Generators to Iteratively pull data (Stop memory overload)
train_generator = train_datagen.flow_from_directory(
                                        '../train/',
                                        target_size=(224,224),
                                        color_mode='rgb',
                                        class_mode='categorical',
                                        batch_size=batch_size
                                    )


test_generator = test_datagen.flow_from_directory(
                                        '../test/',
                                        target_size=(224,224),
                                        color_mode='rgb',
                                        class_mode='categorical',
                                        batch_size=batch_size
                                    )

val_generator = val_datagen.flow_from_directory(
                                        '../valid/',
                                        target_size=(224,224),
                                        color_mode='rgb',
                                        class_mode='categorical',
                                        batch_size=batch_size
                                    )


### Create AlexNet Architecture ###
model = Sequential()

# CONV Layer 1
model.add(Conv2D(filters=96, kernel_size=11, strides=4, padding='same', input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding='valid'))
model.add(BatchNormalization())

# CONV Layer 2
model.add(Conv2D(filters=256, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding='valid'))
model.add(BatchNormalization())

# CONV Layer 3
model.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# CONV Layer 4
model.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.0005)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# CONV Layer 5
model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding='valid'))
model.add(BatchNormalization())

# Flatten Layers
model.add(Flatten())

# FC Layer 1
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.5))

# FC Layer 2
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.6))

# Output Layer
model.add(Dense(units=250, activation='softmax'))

model.summary()

optimizer = keras.optimizers.Adam(lr=0.0003)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

### Callbacks ###
runtime_name = "4CONV_0.6dropout_1024dense_.001L2_{}".format(int(time.time()))
tensorboard =TensorBoard(log_dir='logs/{}'.format(runtime_name),  update_freq='epoch', profile_batch=0)
checkpoints = ModelCheckpoint(filepath=f'checkpoints/4CONV_0.6dropout_1024dense_.001L2_{int(time.time())}.keras', verbose=1, save_best_only=True)
csv_logger = CSVLogger(f'hist/4CONV_0.6dropout_1024dense_.001L2_{int(time.time())}.csv', separator=',', append=False)

hist = model.fit(train_generator,
                 steps_per_epoch=train_generator.n/train_generator.batch_size,
                 validation_data=val_generator,
                 validation_steps=val_generator.n/val_generator.batch_size,
                 epochs=75,
                 callbacks=[tensorboard, checkpoints, csv_logger],
                 verbose=1)
