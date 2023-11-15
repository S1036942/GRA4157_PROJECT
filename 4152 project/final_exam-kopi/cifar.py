from dataloader import DataLoader
import tensorflow as tf

class CIFAR10(DataLoader):
    """
    Class CIFAR10
    A class to load the CIFAR10 dataset, inheriting from DataLoader.

    This class is responsible for loading the CIFAR10 dataset using TensorFlow. 

    Public methods:

    1. loadFunction():
    Loads the CIFAR10 dataset using TensorFlow.
    This method utilizes TensorFlow's keras API to load the CIFAR10 dataset. 
    It is designed to be called during the initialization of the CIFAR10 class.

    @return: The CIFAR10 dataset split into training and test data.
    @rtype: Tuple[ndarray, ndarray]
    """
    def __init__(self):
        super().__init__()
        """
        Initializes the CIFAR10 object by loading the dataset.
        This constructor calls the superclass constructor and then 
        initializes the data loading process by invoking the `loadFunction`.    

        @ivar _data: Stores the loaded CIFAR10 dataset.
        @type _data: Tuple[ndarray, ndarray]
        """
        self._data = self.loadFunction()

    def loadFunction(self):
        """
        Loads the CIFAR10 dataset using TensorFlow.
        This method utilizes TensorFlow's keras API to load the CIFAR10 dataset. 
        It is designed to be called during the initialization of the CIFAR10 class.

        @return: The CIFAR10 dataset split into training and test data.
        @rtype: Tuple[ndarray, ndarray]
        """
        data = tf.keras.datasets.cifar10.load_data()
        return data

    

