from __future__ import print_function, division
import sys
sys.path.append("..")
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet121

# https://keras-cn.readthedocs.io/en/latest/models/model/


class Where_is_Wally():
    def __init__(self):
        # Input shape
        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

    def build_CNN_Network(self):
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

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.img_shape)
        for layer in base_model.layers:
            layer.trainable = False
        d1 = base_model.output
        d2 = flatten(d1)
        d3 = dense(d2, f_size=120, dr=True, lastLayer=False)
        d4 = dense(d3, f_size=84, dr=True, lastLayer=False)
        d5 = dense(d4, f_size=4, dr=False, lastLayer=False)
        #model input output
        return Model(base_model.input, d5)

if __name__ == '__main__':

    # # testing model
    my_CNN = Where_is_Wally()
    my_CNN_Model = my_CNN.build_CNN_Network()
    my_CNN_Model.load_weights('../NR_epoch_.h5')
