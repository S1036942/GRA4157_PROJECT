import tensorflow as tf
from  tensorflow.keras import  layers
from  tensorflow.keras.models  import  Sequential

class NeuralNetwork(tf.keras.Model):
    """
    Class NeuralNetwork
    A superclass for the FullyConNN and ConvNN classes.
    (A subclass of tf.keras.model)

    This class provides a foundational structure for building various types of neural networks. 
    It includes methods for setting hidden layers, classifiers, training, testing, and more. 
    The class is designed to be flexible and extensible for different neural network architectures.

    Public methods:

    1. __repr__():
    A string representation of a NeuralNetwork object.

    2. hidden_layers():
    Abstract method to define hidden layers.
    This method should be implemented in th subclasses of NeuralNetwork to define the hidden layers 
    of the neural network.

    @raise NotImplementedError: When the method is not implemented in a subclass.

    3. train():
    Trains the neural network on a batch of data. Takes the inputs (x and y) and computes the loss for each datapoint. 
    @param inputs the x values and the corresponding y values
    @type tuple consting of all the x values and all the y values 

    @param optimizer The optimizer to use for training.
    @type tf.keras.optimizers.Optimizer

    @return loss the computed loss for the given batch of data, a scalar tensor
    @rtype tf.tensor  (tensorflow.python.framework.ops.EagerTensor)

    4. test():
    Evaluates the neural network on input data 'x' and returns the predicted class labels and the output probabilities.

    @param x the input data to the neural network.
    @type numpy.ndarray

    @return y_hat A tensor of predicted class labels for each input example. 
    @rtype tf.tensor (tensorflow.python.framework.ops.EagerTensor)

    @return pi_hat A tensor representing the output pseudo-probabilities from the classifier layer.
    @rtype tf.tensor (tensorflow.python.framework.ops.EagerTensor)

    5. set_hidden():
    Own method to change the value if the instance variable self._hidden. Is used when we ceate new Instances of FullyConNN and ConvNN.
    In the NeuralNetwork class the self._hidden is None, and is changed to the Sequantial instance when we make FullyConNN and ConvNN objects
    The hidden instance is different for the FullyConNN and ConvNN classes. 
    
    @param new_hidden the intance Sequential which is called _hidden.
    @type Sequential
    
    6. set_cls():
    own method to change the value of the instance variable self._cls. Also called on when we create the FullyConNN and ConvNN objects.
        
    @param new_cls the instance Sequantial which is called cls. 
    @type Sequential

    7. set_params():
    Own method to change the vale of the intance variable self._params. We use it when we create instances of FullyConNN and ConvNN to change the value of self._params
    
    @param new_param A combined list of all trainable variables from both the 'cls' and 'hidden' parts of the neural network.
    @type list
    """
    def __init__(self):
        """
        Initializes an intance of NeuralNetwork.
        Sets up the base state of the neural network, including initializing 
        hidden layers, classifier, and parameters to None. 

        @ivar _neurons The number of neurons
        @type _neuron int or None
        
        @ivar _hidden: The hidden layers of the neural network.
        @type _hidden: tf.keras.layers.Layer or None
    
        @ivar _cls: The classifier or output layer of the neural network.
        @type _cls: tf.keras.layers.Layer or None
        
        @ivar _params: Parameters of the neural network that are trainable.
        @type _params: list or None
        """
        super().__init__()
        self._neurons = None
        self._hidden = None
        self._cls = None
        self._params = None


    def __repr__(self):
        """
        Abstract method for string representation of the NeuralNetwork instance.

        @raise NotImplementedError: Only implemented in the subclasses of NeuralNetwork.
        """
        raise NotImplementedError


    def hidden_layers(self):
        """
        Abstract method to define hidden layers.
        This method should be implemented in th subclasses of NeuralNetwork to define the hidden layers 
        of the neural network.

        @raise NotImplementedError: When the method is not implemented in a subclass.
        """
        raise NotImplementedError


    def _classifier(self, y_dim = 10):
        """
        Constructs the classifier (output layer) of the neural network.

        @param y_dim: The dimensionality of the output layer.
        @type y_dim: int

        @return: cls The classifier layer.
        @rtype: tf.keras.Sequential
        """
        assert self._neurons != None
        cls = Sequential([layers.InputLayer(input_shape=self._neurons),
            layers.Dense(y_dim, activation='softmax')])
        return cls


    def _call(self, inputs):
        """
        Processes the input data through the neural network and calculates the loss.
        @param inputs a tuple of x and y values.
        @type a tuple consisting of the x value (matrix) and the y value (matrix)

        @return loss The loss value computed for the given input..
        @rtype tf.tensor (tensorflow.python.framework.ops.EagerTensor)

        """
        hidden = self._hidden
        assert hidden != None
        cls = self._cls
        x, y = inputs
        out = hidden(x)
        out = cls(out)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, out))
        return loss


    def train(self, inputs, optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)):
        """
        Trains the neural network on a batch of data. Takes the inputs (x and y) and computes the loss for each datapoint. 
        @param inputs the x values and the corresponding y values
        @type tuple consting of all the x values and all the y values 

        @param optimizer The optimizer to use for training.
        @type tf.keras.optimizers.Optimizer

        @return loss the computed loss for the given batch of data, a scalar tensor
        @rtype tf.tensor  (tensorflow.python.framework.ops.EagerTensor)
        """
        params = self._params
        with tf.GradientTape() as tape: 
            loss = self._call(inputs)
        gradients = tape.gradient(loss, params) 
        optimizer.apply_gradients(zip(gradients, params))
        return loss
    
    
    def test(self, x):
        """
        Evaluates the neural network on input data 'x' and returns the predicted class labels and the output probabilities.

        @param x the input data to the neural network.
        @type numpy.ndarray

        @return y_hat A tensor of predicted class labels for each input example. 
        @rtype tf.tensor (tensorflow.python.framework.ops.EagerTensor)

        @return pi_hat A tensor representing the output pseudo-probabilities from the classifier layer.
        @rtype tf.tensor (tensorflow.python.framework.ops.EagerTensor)
        """
        out = self._hidden(x)
        out = self._cls(out)
        pi_hat = out
        y_hat = tf.math.argmax(out ,1)
        return y_hat, pi_hat


    def set_hidden(self, new_hidden):
        """
        Own method to change the value if the instance variable self._hidden. Is used when we ceate new Instances of FullyConNN and ConvNN.
        In the NeuralNetwork class the self._hidden is None, and is changed to the Sequantial instance when we make FullyConNN and ConvNN objects
        The hidden instance is different for the FullyConNN and ConvNN classes. 
        
        @param new_hidden the intance Sequential which is called _hidden.
        @type Sequential
        """
        self._hidden = new_hidden
    
    def set_cls(self, new_cls):
        """
        own method to change the value of the instance variable self._cls. Also called on when we create the FullyConNN and ConvNN objects.
        
        @param new_cls the instance Sequantial which is called cls. 
        @type Sequential
        """
        self._cls = new_cls

    def set_params(self, new_param):
        """
        Own method to change the vale of the intance variable self._params. We use it when we create instances of FullyConNN and ConvNN to change the value of self._params
        
        @param new_param A combined list of all trainable variables from both the 'cls' and 'hidden' parts of the neural network.
        @type list
        """
        self._params = new_param