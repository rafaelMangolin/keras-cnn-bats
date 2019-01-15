from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
import cv2
import numpy as np
import itertools
from sklearn import metrics
import os


K.set_image_dim_ordering('th')

input_size = (160,85)
batch_size = 16

nfolds = {'one':'1','two':'2','three':'3','four':'4','five':'5'}

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

def net0_1(nb_channels, height, width, nb_classes):
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
    print('compiling neural network...')
    model.compile(loss="categorical_crossentropy",
                  optimizer="adadelta",
                  metrics=["accuracy"])
    return model


def run_save_weight(fold):
    print('run net to fold {0}'.format(fold))
    datagen = ImageDataGenerator()
    # test_datagen = ImageDataGenerator()
    train_generator = datagen.flow_from_directory(
            './{0}/Train'.format(fold),  
            batch_size=batch_size,
            target_size=input_size,
            shuffle=False,
            class_mode='categorical')

    validation_generator = datagen.flow_from_directory(
            './{0}/Test'.format(fold),  
            batch_size=batch_size,
            shuffle=False,
            target_size=input_size,
            class_mode='categorical')

    net = net0(3,input_size[0],input_size[1],4)
    net2 = net0_1(3,input_size[0],input_size[1],4)
    # print(net.summary())
    net.load_weights("fold_{0}_weight.h5".format(fold))
    net2.load_weights("fold_{0}_weight.h5".format(fold),by_name=True)
    
    results_c = []
    results_p = []
    labels = []
    for i in range(50):
        x,y = next(validation_generator)
        c = net.predict_classes(x,batch_size=batch_size,verbose=0)
        p = net.predict(x,batch_size=batch_size,verbose=0)
        results_p = itertools.chain(results_p , p)
        results_c = itertools.chain(results_c , c)
        labels = itertools.chain(labels , y)

    results_p = list(results_p)
    results_c = list(results_c)
    labels = list(labels)
    names = list(validation_generator.filenames)

    result = []

    features = list(net2.predict_generator(train_generator,3200))
    namesf = list(train_generator.filenames)
    features = list(itertools.chain(features,net2.predict_generator(validation_generator,800)))
    namesf = list(itertools.chain(namesf,validation_generator.filenames))

    for i in range(800):
        arr = correctOrder(results_p[i])
        arr = ' '.join(map(str,arr))
        name = clearName(names[i])
        result.append(name+' '+arr)
        
    f = open("F{0}.prediction".format(nfolds[fold]),"w+")
    f.write('\n'.join(result))

    result = []
    for i in range(4000):
        arr = list(features[i])
        arr = ';'.join(map(str,arr))
        name = clearName(namesf[i])
        result.append(name+';'+arr)

    f = open("F{0}_L5.features".format(nfolds[fold]),"w+")
    f.write('\n'.join(result))
    

def correctOrder(arr):
    return [arr[1],arr[3],arr[2],arr[0]]

def clearName(name):
    name = name.replace('four/four.','')
    name = name.replace('one/one.','')
    name = name.replace('two/two.','')
    name = name.replace('three/three.','')
    return name
    
run_save_weight('one');
run_save_weight('two');
run_save_weight('three');
run_save_weight('four');
run_save_weight('five');
