import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, AveragePooling2D

# Initializer for kernel and bias
kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

# Feed an input X into multiple convolutions of different sizes
def inception(x, filter1, filter3_r, filter3, filter5_r, filter5, filter_pool, name=None):
    # 1 X 1 Convolution
    conv1 = Conv2D(filter1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                   bias_initializer=bias_init)(x)

    # 3 X 3 Convolution/ Dimensionality reduction 1X1 Layer
    conv3 = Conv2D(filter3_r, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                   bias_initializer=bias_init)(x)
    conv3 = Conv2D(filter3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init,
                   bias_initializer=bias_init)(conv3)

    # 5 X 5 Convolution/ Dimensionality reduction 1X1 Layer
    conv5 = Conv2D(filter5_r, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                   bias_initializer=bias_init)(x)
    conv5 = Conv2D(filter5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init,
                   bias_initializer=bias_init)(conv5)

    # Pooling Layer
    pool_proj = MaxPool2D(pool_size=3, strides=1, padding='same')(x)
    pool_proj = Conv2D(filter_pool, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                       bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv1, conv3, conv5, pool_proj], name=name)

    return output



input_layer = Input(shape=(224, 224, 3))

x = Conv2D(filters=64, kernel_size=76, padding='same', strides=2, activation='relu', name='conv_1_7x7',
           kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x = MaxPool2D(pool_size=3, padding='same', strides=2, name='max_pool_1_3x3')(x)
x = Conv2D(192, (3,3), padding='same', strides=1, activation='relu', name='conv_2_3x3')(x)
x = MaxPool2D(pool_size=3, padding='same', strides=2, name='max_pool_2_3x3')(x)

### 2X Inception Modules + MaxPooling ###
x = inception(x,
              filter1=64,
              filter3_r=96,
              filter3=128,
              filter5_r=16,
              filter5=32,
              filter_pool=32,
              name='inception_1a')

x = inception(x,
              filter1=128,
              filter3_r=128,
              filter3=192,
              filter5_r=32,
              filter5=96,
              filter_pool=64,
              name='inception_1b')

x = MaxPool2D(pool_size=3, padding='same', strides=2, name='max_pool_3_3x3')(x)

### 5X Inception Modules + MaxPooling ###
x = inception(x,
              filter1=192,
              filter3_r=96,
              filter3=208,
              filter5_r=16,
              filter5=48,
              filter_pool=64,
              name='inception_2a')


x = inception(x,
              filter1=160,
              filter3_r=112,
              filter3=224,
              filter5_r=24,
              filter5=64,
              filter_pool=64,
              name='inception_2b')

x = inception(x,
              filter1=128,
              filter3_r=128,
              filter3=256,
              filter5_r=24,
              filter5=64,
              filter_pool=64,
              name='inception_2c')

x = inception(x,
              filter1=112,
              filter3_r=144,
              filter3=288,
              filter5_r=32,
              filter5=64,
              filter_pool=64,
              name='inception_2d')


x = inception(x,
              filter1=256,
              filter3_r=160,
              filter3=320,
              filter5_r=32,
              filter5=128,
              filter_pool=128,
              name='inception_2e')

x = MaxPool2D(pool_size=3, padding='same', strides=2, name='max_pool_4_3x3')(x)

### 2X inception modules ###

x = inception(x,
              filter1=256,
              filter3_r=160,
              filter3=320,
              filter5_r=32,
              filter5=128,
              filter_pool=128,
              name='inception_3a')

x = inception(x,
              filter1=384,
              filter3_r=192,
              filter3=384,
              filter5_r=48,
              filter5=128,
              filter_pool=128,
              name='inception_3b')


### Classifier ###

x = AveragePooling2D(pool_size=7, strides=1, padding='valid', name='avg_pool_classifier')(x)

x = Dropout(0.4)(x)

x = Dense(1000, activation='relu', name='linear')(x)
x = Dense(1000, activation='softmax', name='output')(x)

model = Model(input_layer, x, name='GoogleNet')
model.summary()


