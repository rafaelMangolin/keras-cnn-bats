from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras import backend as K
K.set_image_dim_ordering('th')


def net0(nb_channels, height, width, nb_classes):
    print('creating neural network...')
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu',
                            input_shape=(nb_channels, height, width)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    print('compiling neural network...')
    model.compile(loss="categorical_crossentropy",
                  optimizer="adadelta",
                  metrics=["accuracy"])
    return model


def net1(nb_channels, height, width, nb_classes):
    print('creating neural network...')
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu',
                            input_shape=(nb_channels, height, width)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same',
                            input_shape=(nb_channels, height, width)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same',
                            input_shape=(nb_channels, height, width)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    print('compiling neural network...')
    model.compile(loss="categorical_crossentropy",
                  optimizer="adadelta",
                  metrics=["accuracy"])
    return model


def test(nb_channels, height, width, nb_classes):
    print('creating testing neural network...')
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu',
                            input_shape=(nb_channels, height, width)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(nb_channels, activation='relu'))
    model.add(Dense(nb_channels, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    print('compiling neural network...')
    model.compile(loss="categorical_crossentropy",
                  optimizer="adadelta",
                  metrics=["accuracy"])
    return model
