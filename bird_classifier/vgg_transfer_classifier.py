### IMPORT DATA ###
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend
from keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from keras.models import Sequential
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


# Pull data and normalize image values
train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=30,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=False
                                   )
test_datagen = ImageDataGenerator(rescale=1/255,
                                  rotation_range=30,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  horizontal_flip=True,
                                  vertical_flip=False
                                  )
val_datagen = ImageDataGenerator(rescale=1/255,
                                 rotation_range=30,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=False
                                 )



# Generators to Iteratively pull data (Stop major memory overload)
train_generator= train_datagen.flow_from_directory(
    'train/',
    target_size=(224,224),
    color_mode='rgb',
    class_mode = 'categorical',
    batch_size=256
)

test_generator= test_datagen.flow_from_directory(
    'test/',
    target_size=(224,224),
    color_mode='rgb',
    class_mode = 'categorical',
    batch_size=256
)

val_generator= val_datagen.flow_from_directory(
    'valid/',
    target_size=(224,224),
    color_mode='rgb',
    class_mode = 'categorical',
    batch_size=256
)


# Get list of all unique indexes
raw_bird_dict = train_generator.class_indices
bird_list = {v:k.lower().title() for k, v in raw_bird_dict.items()}

backend.clear_session()
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3)) # Dont include FC Layers

# Make only the last 6 layers trainable
for layer in vgg.layers[:-6]:
    layer.trainable = False

### Built Network Architecture based on VGGNET ###
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(250, activation='softmax'))

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(train_generator,
                 steps_per_epoch=train_generator.n/train_generator.batch_size,
                 validation_data=val_generator,
                 validation_steps=val_generator.n/val_generator.batch_size,
                 epochs=2,
                 verbose=1)

