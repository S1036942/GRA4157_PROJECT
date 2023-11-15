import tensorflow as tf
from  tensorflow.keras import  layers
from  tensorflow.keras.models  import  Sequential

class NeuralNetwork(tf.keras.Model):
    """
    A base class for creating neural network models using TensorFlow.

    This class provides a foundational structure for building various types of neural networks. 
    It includes methods for setting hidden layers, classifiers, training, testing, and more. 
    The class is designed to be flexible and extensible for different neural network architectures.

    @ivar _hidden: The hidden layers of the neural network.
    @type _hidden: tf.keras.layers.Layer or None
    @ivar _cls: The classifier or output layer of the neural network.
    @type _cls: tf.keras.layers.Layer or None
    @ivar _params: Parameters of the neural network that are trainable.
    @type _params: list or None
    @ivar _last_layer_neurons: Number of neurons in the last layer of the neural network.
    @type _last_layer_neurons: int
    """
    def __init__(self):
        """
        Initializes the NeuralNetwork instance.

        Sets up the base state of the neural network, including initializing 
        hidden layers, classifier, and parameters to None. Also sets the number 
        of neurons in the last layer to 0.
        """
        super().__init__()
        self._hidden = None
        self._cls = None
        self._params = None

        self._last_layer_neurons = 0

    def __repr__(self):
        """
        Abstract method for string representation of the NeuralNetwork instance.

        @raise NotImplementedError: When the method is not implemented in a subclass.
        """
        #print("<NeuralNetwork>")
        raise NotImplementedError

    def hidden_layers(self):
        """
        Abstract method to define hidden layers.

        This method should be implemented in subclasses to define the hidden layers 
        of the neural network.

        @raise NotImplementedError: When the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def classifier(self, neurons = 50, y_dim = 10):
        """
        Constructs the classifier (output layer) of the neural network.

        @param neurons: The number of neurons in the input to the classifier.
        @type neurons: int
        @param y_dim: The dimensionality of the output layer.
        @type y_dim: int

        @return: The classifier layer.
        @rtype: tf.keras.Sequential
        """
        cls = Sequential([layers.InputLayer(input_shape=neurons),
            layers.Dense(y_dim, activation='softmax')])
        return cls

    def call(self, x, y):
        """
        Processes the input through the neural network and computes loss.

        @param x: Input data.
        @type x: tensor
        @param y: True labels for the input data.
        @type y: tensor

        @return: The loss value computed for the given input.
        @rtype: tensor
        """
        hidden = self._hidden
        cls = self._cls
        
        out = hidden(x)
        out = cls(out)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, out))
        return loss

    def train(self, inputs, optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)):
        """
        Trains the neural network using the provided inputs and optimizer.

        @param inputs: Input data for training.
        @type inputs: tensor
        @param optimizer: The optimizer to use for training.
        @type optimizer: tf.keras.optimizers.Optimizer

        @return: The loss after training on the input data.
        @rtype: tensor
        """
        params = self._params
        with tf.GradientTape() as tape: 
            loss = self.call(inputs)
        gradients = tape.gradient(loss, params) 
        optimizer.apply_gradients(zip(gradients, params))
        return loss
    
    
    def test(self, x):
        """
        Tests the neural network on the provided input data.

        @param x: Input data for testing.
        @type x: tensor

        @return: Predicted labels and prediction probabilities for the input data.
        @rtype: (tensor, tensor)
        """
        out = self._hidden(x)
        out = self._cls(out)
        pi_hat = out
        y_hat = tf.math.argmax(out ,1)
        return y_hat, pi_hat # WE need pi_hat for AUC

    def set_hidden(self, new_hidden):
        """
        Sets the hidden layers of the neural network.

        @param new_hidden: The new hidden layers to set.
        @type new_hidden: tf.keras.layers.Layer
        """
        self._hidden = new_hidden
    
    def set_cls(self, new_cls):
        """
        Sets the classifier (output layer) of the neural network.

        @param new_cls: The new classifier to set.
        @type new_cls: tf.keras.layers.Layer
        """
        self._cls = new_cls

    def set_params(self, new_param):
        """
        Sets the trainable parameters of the neural network.

        @param new_param: The new parameters to set.
        @type new_param: list
        """
        self._params = new_param