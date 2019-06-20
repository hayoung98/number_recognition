import numpy as np
import scipy.misc
import imageio
import cv2
import csv

class DataLoader():
    def __init__(self, dataset_name, img_res=96):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.paths = []
        self.values = []

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"

        with open(self.dataset_name + "/" + data_type + ".csv", "r") as file:
            self.paths=[]
            self.values = []
            file.seek(0)
            imgs = []
            reader = csv.reader(file, delimiter=",")
            for index, row in enumerate(reader):
                self.paths.append(row[0])
                self.values.append(int(row[1]))

        self.values = np.array(self.values)
        self.n_batches = int(len(self.paths) / batch_size)

        index = [i for i in range(len(self.paths))]
        np.random.shuffle(index)

        for idx in range(self.n_batches - 1):
            img = imgs[idx * batch_size:(idx + 1) * batch_size]
            values = self.values[idx * batch_size:(idx + 1) * batch_size]
            r_batch = index[idx * batch_size:(idx + 1) * batch_size]

            imgs, labels =[],[]
            for j in r_batch:
                imgs.append(self.imread(self.paths[j]))
                value_ohe = self.one_hot_encode(self.values[j],num_classes=10)
                labels.append(value_ohe)

            imgs = np.array(imgs) / 127.5 - 1.
            labels = np.array(labels)

            yield imgs, labels

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"

        with open(self.dataset_name + "/" + data_type + ".csv", "r") as file:
            self.paths = []
            self.values = []
            file.seek(0)
            reader = csv.reader(file, delimiter=",")
            for index, row in enumerate(reader):
                self.paths.append(row[0])
                self.values.append(int(row[1]))

        self.values = np.array(self.values)
        index = [i for i in range(len(self.paths))]
        np.random.shuffle(index)
        batch_imgs = np.random.choice(index, size=batch_size)
        imgs, labels = [], []
        """        
        indices = (len(self.values)*np.random.rand(batch_size)).astype(int)
        img = imgs[indices]
        img_n = np.array(img) / 127.5 - 1.
        """

        for j in range(batch_size):
            imgs.append(self.imread(self.paths[batch_imgs[j]]))
            value_ohe = self.one_hot_encode(self.values[batch_imgs[j]], num_classes=10)
            labels.append(value_ohe)

        imgs = np.array(imgs) / 127.5 - 1.
        labels = np.array(labels)

        return imgs, labels

    def imread(self, path):
        return scipy.misc.imread(path).astype(np.float)

    def one_hot_encode(self, y, num_classes=0):
        return np.squeeze(np.eye(num_classes)[y.reshape(-1)])