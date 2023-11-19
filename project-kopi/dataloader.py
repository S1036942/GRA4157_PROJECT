import tensorflow as tf
import numpy as np

class DataLoader:
    """
    Class DataLoader:
    A base class for data loading functionalities in TensorFlow models.

    This class provides a framework for loading and preprocessing data, 
    with methods for creating TensorFlow data loaders and properties 
    to access processed training and test data.

    Public methods:
    
    1. loadFunction():
    Abstract method to load data.
    This method is implemented in subclasses to define how data is loaded.

    2 loader():
    Creates a TensorFlow data loader object.
    This method preprocesses the data and creates a TensorFlow data loader using the 
    training data. It allows shuffling and batching of the data.

    @param batch_size: The size of each batch of data.
    @type batch_size: int

    @return: A TensorFlow data loader object.
    @rtype: tf.data.Dataset

    3. x_tr():
    Retrieves the training data (features) from _data.
    Converts the values to a number between 0-1.

    @return: The normalized training feature data.
    @rtype: np.ndarray

    4. x_te():
    Retrieves the test data (features) from _data.
    Converts the values to a number between 0-1.

    @return: The normalized test feature data.
    @rtype: np.ndarray

    5. y_tr():
    Retrieves the training data (labels) from _data.
    This property processes the training labels data.

    @return: The processed training labels data.
    @rtype: np.ndarray

    6 y_te():
    Retrieves the test data (labels) from _data.
    This property processes the test labels data.

    @return: The processed test labels data.
    @rtype: np.ndarray
    """
    def __init__(self):
        """
        Initializes the DataLoader instance.
        Sets the initial state of the data attribute to None.

        @ivar _data: A placeholder for storing the loaded data.
        @type _data: tuple or None
        """
        self._data = None
    
    def loadFunction(self):
        """
        Abstract method to load data.
        This method is  implemented in subclasses to define how data is loaded.

        @raise NotImplementedError: When the method is not implemented in a subclass.
        """
        raise NotImplementedError
 
    def loader(self, batch_size):
        """
        Creates a TensorFlow data loader object.
        This method preprocesses the data and creates a TensorFlow data loader using the 
        training data. It allows shuffling and batching of the data.

        @param batch_size: The size of each batch of data.
        @type batch_size: int

        @return: A TensorFlow data loader object.
        @rtype: tf.data.Dataset
        """
        tf_dl = tf.data.Dataset.from_tensor_slices((self.x_tr,self.y_tr)).shuffle(self.x_tr.shape[0]).batch(batch_size) 
        return tf_dl

    @property
    def x_tr(self):
        """
        Retrieves the training data (features) from _data.
        Converts the values to a number between 0-1.

        @return: The normalized training feature data.
        @rtype: np.ndarray
        """
        return np.float32(np.array(self._data[0][0])) / 255
    
    @property
    def x_te(self):
        """
        Retrieves the test data (features) from _data.
        Converts the values to a number between 0-1.

        @return: The normalized test feature data.
        @rtype: np.ndarray
        """
        return np.float32(np.array(self._data[1][0])) / 255
    
    @property
    def y_tr(self):
        """
        Retrieves the training data (labels) from _data.
        This property processes the training labels data.

        @return: The processed training labels data.
        @rtype: np.ndarray
        """
        return tf.keras.utils.to_categorical(self._data[0][1])
    
    @property
    def y_te(self):
        """
        Retrieves the test data (labels) from _data.
        This property processes the test labels data.

        @return: The processed test labels data.
        @rtype: np.ndarray
        """
        return self._data[1][1]
       
        
    

    