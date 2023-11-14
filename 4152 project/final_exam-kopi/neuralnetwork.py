import tensorflow as tf
from  tensorflow.keras import  layers
from  tensorflow.keras.models  import  Sequential

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._hidden = None
        self._cls = None
        self._params = None

        self._last_layer_neurons = 0

    def __repr__(self):
        #print("<NeuralNetwork>")
        raise NotImplementedError

    def hidden_layers(self):
        raise NotImplementedError

    def classifier(self, neurons = 50, y_dim = 10):
        cls = Sequential([layers.InputLayer(input_shape=neurons),
            layers.Dense(y_dim, activation='softmax')])
        return cls

    def call(self, x, y):
        hidden = self._hidden
        cls = self._cls
        
        out = hidden(x)
        out = cls(out)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, out))
        return loss

    def train(self, inputs, optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)):
        params = self._params
        with tf.GradientTape() as tape: 
            loss = self.call(inputs)
        gradients = tape.gradient(loss, params) 
        optimizer.apply_gradients(zip(gradients, params))
        return loss
    
    
    def test(self, x):
        out = self._hidden(x)
        out = self._cls(out)
        pi_hat = out
        y_hat = tf.math.argmax(out ,1)
        return y_hat, pi_hat # WE need pi_hat for AUC

    def set_hidden(self, new_hidden):
        self._hidden = new_hidden
    
    def set_cls(self, new_cls):
        self._cls = new_cls

    def set_params(self, new_param):
        self._params = new_param