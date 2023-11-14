from dataloader import DataLoader
import tensorflow as tf

class CIFAR10(DataLoader):
    def __init__(self):
        super().__init__()
        self._data = self.loadFunction()

    def loadFunction(self):
        data = tf.keras.datasets.cifar10.load_data()
        return data

    

