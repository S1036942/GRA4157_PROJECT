from dataloader import DataLoader
import tensorflow as tf
import numpy as np

class MNIST(DataLoader):
    def __init__(self):
        super().__init__()
        self._data = self.loadFunction()

    def loadFunction(self):
        data = tf.keras.datasets.mnist.load_data(path="mnist.npz")
        return data
    
    @property
    def x_tr(self):
       return np.reshape(super().x_tr, (60000, 782))
    
    @property
    def x_te(self):
        return np.reshape(super().x_te, (10000, 782))
