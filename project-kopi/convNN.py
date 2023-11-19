from neuralnetwork import NeuralNetwork
from  tensorflow.keras import  layers
from  tensorflow.keras.models  import  Sequential

class ConvNN(NeuralNetwork):
    """
    Class ConvNN
    A subclass of NeuralNetwork that implements a convolutional neural network (ConvNN).
    This class creates a convolutional neural network using the TensorFlow Keras API. It allows 
    for the specification of various parameters such as the number of neurons, filters, 
    kernel size, and strides in the hidden layers of the network.

    Public methods:

    1. __repr__():
        Represents the ConvNN instance as a string. Prints out the name of the class

    2. hidden_layers():
    Creates the hidden layers of the ConvNN. and calls the _classifier method.

    This method constructs the hidden layers using convolutional layers. The parameters for 
    the layers such as the number of neurons, filters, kernel size, and strides can be 
    specified.

    @param input_shape: The shape of the input data.
    @type input_shape: tuple

    @param neurons: The number of neurons in the final convolutional layer.
    @type neurons: int

    @param filters: The number of filters in the convolutional layers.
    @type filters: int

    @param kernel_size: The size of the kernel in the convolutional layers.
    @type kernel_size: int

    @param strides: The stride size in the convolutional layers.
    @type strides: tuple

    @return: The constructed hidden layers of the ConvNN.
    @rtype: Sequential
    """
    def __init__(self):
        """
        Initializes the ConvNN object.
        This constructor initializes the ConvNN by setting up the hidden layers, 
        classifier, and parameters of the neural network.
        """
        super().__init__()
        self.set_hidden(self.hidden_layers())
        self.set_params(self._cls.trainable_variables + self._hidden.trainable_variables)

    def __repr__(self):
        """
        Represents the ConvNN instance as a string. Prints out then name of the class.
        """
        print("<Convolutional NeuralNetwork>")


    def hidden_layers(self, input_shape = (32, 32, 3), neurons = 50, filters = 32, kernel_size = 3, strides = (2, 2)):
        """
        Creates the hidden layers of the ConvNN. and calls the _classifier method.

        This method constructs the hidden layers using convolutional layers. The parameters for 
        the layers such as the number of neurons, filters, kernel size, and strides can be 
        specified.

        @param input_shape: The shape of the input data.
        @type input_shape: tuple

        @param neurons: The number of neurons in the final convolutional layer.
        @type neurons: int

        @param filters: The number of filters in the convolutional layers.
        @type filters: int

        @param kernel_size: The size of the kernel in the convolutional layers.
        @type kernel_size: int

        @param strides: The stride size in the convolutional layers.
        @type strides: tuple

        @return: The constructed hidden layers of the ConvNN.
        @rtype: Sequential
        """
        hidden = Sequential([layers.InputLayer(input_shape=input_shape)
                ,layers.Conv2D(filters=filters ,kernel_size=kernel_size ,strides=strides)
                ,layers.Conv2D(filters =2* filters ,kernel_size=kernel_size ,strides=strides)
                ,layers.Conv2D(filters=neurons ,kernel_size=kernel_size ,strides =(5 ,5))
                ,layers.Flatten ()])
        self._neurons = neurons
        self.set_cls(self._classifier())
        return hidden
    
    
        

