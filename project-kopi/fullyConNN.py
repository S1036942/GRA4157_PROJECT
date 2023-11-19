from neuralnetwork import NeuralNetwork
from  tensorflow.keras import  layers
from  tensorflow.keras.models  import  Sequential

class FullyConNN(NeuralNetwork):
    """
    Class FullyConNN
    A subclass of NeuralNetwork that implements a fully connected neural network (FullyConNN).

    This class creates a fully connected neural network using the TensorFlow Keras API. It provides 
    functionalities to set up hidden layers and a classifier for the network.

    Public methods:

    1. __repr__():
        nRepresents the FullyConNN instance as a string. Prints out then name of the class.

    2. hidden_layers():
    Creates the hidden layers of the FullyConNN.

    This method constructs the hidden layers using dense layers. The parameters for 
    the layers such as the input shape and the number of neurons can be specified.

    @param input_shape: The number of features in the input data. Default 28*28
    @type input_shape: int

    @param neurons: The number of neurons in each dense layer. Default 50
    @type neurons: int

    @return: The constructed hidden layers of the FullyConNN.
    @rtype: Sequential

    """
    def __init__(self):
        """
        Initializes the FullyConNN object.
        This constructor initializes the FullyConNN by setting up the hidden layers, 
        classifier, and parameters of the neural network.
        """
        super().__init__()
        self.set_hidden(self.hidden_layers())
        self.set_params(self._cls.trainable_variables + self._hidden.trainable_variables)

    def __repr__(self):
        """
        Represents the FullyConNN instance as a string. Prints out then name of the class.
        """
        return print("<Fully Connencted NeuralNetwork>")

    def hidden_layers(self, input_shape = 28*28, neurons = 50):
        """
        Creates the hidden layers of the FullyConNN. And calls the _classifier method. 

        This method constructs the hidden layers using dense layers. The parameters for 
        the layers such as the input shape and the number of neurons can be specified.

        @param input_shape: The number of features in the input data. Default 28*28
        @type input_shape: int

        @param neurons: The number of neurons in each dense layer. Default 50
        @type neurons: int

        @return: The constructed hidden layers of the FullyConNN.
        @rtype: Sequential
        """
        hidden = Sequential([layers.InputLayer(input_shape=input_shape)
                , layers.Dense(neurons)
                , layers.Dense(neurons)])
        self._neurons = neurons
        self.set_cls(self._classifier())
        return hidden