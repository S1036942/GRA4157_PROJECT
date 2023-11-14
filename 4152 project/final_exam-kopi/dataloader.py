import tensorflow as tf
import numpy as np

class DataLoader:
    def __init__(self):
        self._data = None
    
    def loadFunction(self):
        raise NotImplementedError
 
    ###
    # @x: numpy array
    # @y: numpy array
    # @return: tf data loader object
    def loader(self, batch_size):
        tf_dl = tf.data.Dataset.from_tensor_slices((self.x_tr,self.y_tr)).shuffle
        (self.x_tr.shape[0]).batch(batch_size) 
        return tf_dl

    @property
    def x_tr(self):
        return np.float32(np.array(self._data[0][0])) / 255
    
    @property
    def x_te(self):
        return np.float32(np.array(self._data[1][0])) / 255
    
    @property
    def y_tr(self):
        return self._data[0][1].tf.keras.utils.to_categorical()
    
    @property
    def y_te(self):
        return self._data[1][1].tf.keras.utils.to_categorical()
       
        
    

    