from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import tensorflow as tf
import datetime
from data_loader import DataLoader
import numpy as np
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.applications.densenet import DenseNet121
import scipy.misc
import cv2 as cv2


class Number_Recognition():
    def __init__(self):
        # Input shape
        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = './number_data'
        self.data_loader = DataLoader( dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))
        # Build the network
        self.HUBER_DELTA = 0.5
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.CNN_Network = self.build_CNN_Network()
        self.CNN_Network.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Input images
        Xtr = Input(shape=self.img_shape)

        # Output coords
        Ycoords = self.CNN_Network(Xtr)

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

        return Model(base_model.input, d5)

    #MSE
    """
    def smoothL1(self, y_true, y_pred):
        x = K.abs(y_true - y_pred)
        x = K.switch(x < self.HUBER_DELTA, 0.5 * x ** 2, self.HUBER_DELTA * (x - 0.5 * self.HUBER_DELTA))
        return K.sum(x)
    """
    def smoothL1(self, y_true, y_pred):
        x = K.abs(y_true - y_pred)
        x = x ** 2
        return K.sum(x)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, (imgs, values) in enumerate(self.data_loader.load_batch(batch_size, is_testing=False)):
                #  Training
                crossentropy_loss = self.CNN_Network.train_on_batch(imgs, values)

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [Training loss: %f, Training mse: %f] time: %s" % (
                    epoch + 1, epochs,
                    batch_i + 1, self.data_loader.n_batches - 1,
                    crossentropy_loss[0], 100 * crossentropy_loss[1],
                    elapsed_time))
                # If at save interval => do validation and save model
                if (batch_i + 1) % sample_interval == 0:
                    self.validation(epoch, batch_i + 1)

    def validation(self, epoch, num_batch):
        imgs, values = self.data_loader.load_data(batch_size=1, is_testing=True)
        pred_values = self.CNN_Network.predict(imgs)
        print("Validation acc: " + str(
            int(accuracy_score(np.argmax(values, axis=1), np.argmax(pred_values, axis=1)) * 100)) + "%")

        if num_batch == self.data_loader.n_batches - 1:
            self.CNN_Network.save('./saved_model/NR_epoch_%d.h5' % epoch)


if __name__ == '__main__':
    # # training model
    my_CNN_Model = Number_Recognition()
    my_CNN_Model.train(epochs=100, batch_size=21, sample_interval=1265)