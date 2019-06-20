from __future__ import print_function, division
import sys
sys.path.append("..")
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import cv2
import numpy as np
import scipy.misc


# https://keras-cn.readthedocs.io/en/latest/models/model/


class Number_Recognition():
    def __init__(self):
        # Input shape
        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

    def build_CNN_Network(self):

        def conv2d(layer_input, filters, f_size=4, stride=2, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='valid')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def maxpooling2d(layer_input, f_size, stride=2):
            d = MaxPooling2D(pool_size=f_size, strides=stride, padding='valid')(layer_input)
            return d

        def flatten(layer_input):
            d = Flatten()(layer_input)
            return d

        def dense(layer_input, f_size, dr=True, lastLayer=True):
            if lastLayer:
                d = Dense(f_size, activation='softmax')(layer_input)
            else:
                d = Dense(f_size, activation='linear')(layer_input)
                d = LeakyReLU(alpha=0.2)(d)
                if dr:
                    d = Dropout(0.5)(d)
            return d

        # LeNet-5 layers
        d0 = Input(shape=self.img_shape) # Image input
        d1 = conv2d(d0, filters=6, f_size=5, stride=1, bn=True)
        d2 = maxpooling2d(d1, f_size=2, stride=2)
        d3 = conv2d(d2, filters=16, f_size=5, stride=1, bn=True)
        d4 = maxpooling2d(d3, f_size=2, stride=2)
        d5 = flatten(d4)
        d6 = dense(d5, f_size=120, dr=True, lastLayer=False)
        d7 = dense(d6, f_size=84, dr=True, lastLayer=False)
        d8 = dense(d7, f_size=10, dr=False, lastLayer=True)
        #d9 = dense(d8, f_size=2, dr=False, lastLayer=True)

        return Model(d0, d8)

if __name__ == '__main__':

    # # testing model
    my_CNN = Number_Recognition()
    my_CNN_Model = my_CNN.build_CNN_Network()
    my_CNN_Model.load_weights('./saved_model/NR_epoch_5.h5')
    filename = "./number_data/test_img/0 (1).jpg"
    img = cv2.imread(filename).astype(np.float)
    img = np.array(img) / 127.5 - 1.

    print(my_CNN_Model.predict(img))