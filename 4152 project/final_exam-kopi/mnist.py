from dataloader import DataLoader
import tensorflow as tf
import numpy as np

class MNIST(DataLoader):
    """
    Class MNIST
    A subclass of DataLoader specialized for loading and processing the MNIST dataset.

    This class extends the functionality of DataLoader to specifically handle the MNIST dataset. 
    It includes methods to load the dataset and reshape the feature data for training and testing.

    Public methods:

    1. loadFunction():
    Loads the MNIST dataset using TensorFlow.
    This method utilizes TensorFlow's keras API to load the MNIST dataset. It is designed 
    to be called during the initialization of the MNIST class.

    @return: The MNIST dataset split into training and test data.
    @rtype: tuple

    2. x_tr():
    Retrieves and reshapes the training data (features) from _data.
    This property processes the training feature data by reshaping it to the 
    desired format for model training. In this case (10000, 784)

    @return: The reshaped training feature data.
    @rtype: np.ndarray

    3. x_te():
    Retrieves and reshapes the test data (features) from _data. 
    This property processes the test feature data by reshaping it to the 
    desired format for model evaluation. In this case (10000, 784)

    @return: The reshaped test feature data.
    @rtype: np.ndarray

    """
    def __init__(self):
        """
        Initializes the MNIST object by loading the dataset.
        This constructor calls the superclass constructor and then initializes the data 
        loading process by invoking the `loadFunction`.

        @ivar _data: Stores the loaded MNIST dataset.
        @type _data: tuple or None
        """
        super().__init__()
        self._data = self.loadFunction()

    def loadFunction(self):
        """
        Loads the MNIST dataset using TensorFlow.
        This method utilizes TensorFlow's keras API to load the MNIST dataset. It is designed 
        to be called during the initialization of the MNIST class.

        @return: The MNIST dataset split into training and test data.
        @rtype: tuple
        """
        data = tf.keras.datasets.mnist.load_data(path="mnist.npz")
        return data
    
    @property
    def x_tr(self):
        """
        Retrieves and reshapes the training data (features) from _data.
        This property processes the training feature data by reshaping it to the 
        desired format for model training. In this case (10000, 784)

        @return: The reshaped training feature data.
        @rtype: np.ndarray
        """
        return np.reshape(super().x_tr, (60000, 784))
    
    @property
    def x_te(self):
        """
        Retrieves and reshapes the test data (features) from _data. 
        This property processes the test feature data by reshaping it to the 
        desired format for model evaluation. In this case (10000, 784)

        @return: The reshaped test feature data.
        @rtype: np.ndarray
        """
        return np.reshape(super().x_te, (10000, 784))
