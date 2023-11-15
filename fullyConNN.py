from neuralnetwork import NeuralNetwork
from  tensorflow.keras import  layers
from  tensorflow.keras.models  import  Sequential

class FullyConNN(NeuralNetwork):
    """
    A subclass of NeuralNetwork that implements a fully connected neural network (FullyConNN).

    This class creates a fully connected neural network using the TensorFlow Keras API. It provides 
    functionalities to set up hidden layers and a classifier for the network.

    @ivar _last_layer_neurons: Stores the number of neurons in the last layer of the hidden layers.
    @type _last_layer_neurons: int
    """
    def __init__(self):
        """
        Initializes the FullyConNN object.

        This constructor initializes the FullyConNN by setting up the hidden layers, 
        classifier, and parameters of the neural network.
        """
        super().__init__()
        self.set_hidden(self.hidden_layers())
        self.set_cls(self.classifier())
        self.set_params(self._cls.trainable_variables + self._hidden.trainable_variables)

    def __repr__(self):
        """
        Represents the FullyConNN instance as a string.

        @return: A string representation of the FullyConNN instance.
        @rtype: str
        """
        return print("<Fully Connencted NeuralNetwork>")

    def hidden_layers(self, input_shape = 28*28, neurons = 50):
        
class FullyConNN(NeuralNetwork):
    """
    A subclass of NeuralNetwork that implements a fully connected neural network (FullyConNN).

    This class creates a fully connected neural network using the TensorFlow Keras API. It provides 
    functionalities to set up hidden layers and a classifier for the network.

    @ivar _last_layer_neurons: Stores the number of neurons in the last layer of the hidden layers.
    @type _last_layer_neurons: int
    """

    def __init__(self):
        """
        Initializes the FullyConNN object.

        This constructor initializes the FullyConNN by setting up the hidden layers, 
        classifier, and parameters of the neural network.
        """
        super().__init__()
        self.set_hidden(self.hidden_layers())
        self.set_cls(self.classifier())
        self.set_params(self._cls.trainable_variables + self._hidden.trainable_variables)

    def __repr__(self):
        """
        Represents the FullyConNN instance as a string.

        @return: A string representation of the FullyConNN instance.
        @rtype: str
        """
        return "<Fully Connected NeuralNetwork>"

    def hidden_layers(self, input_shape=28*28, neurons=50):
        """
        Creates the hidden layers of the FullyConNN.

        This method constructs the hidden layers using dense layers. The parameters for 
        the layers such as the input shape and the number of neurons can be specified.

        @param input_shape: The number of features in the input data.
        @type input_shape: int
        @param neurons: The number of neurons in each dense layer.
        @type neurons: int

        @return: The constructed hidden layers of the FullyConNN.
        @rtype: Sequential
        """
        hidden = Sequential([layers.InputLayer(input_shape=input_shape)
                , layers.Dense(neurons)
                , layers.Dense(neurons)])
        self._last_layer_neurons = neurons
        return hidden