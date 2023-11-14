from neuralnetwork import NeuralNetwork
from  tensorflow.keras import  layers
from  tensorflow.keras.models  import  Sequential

class FullyConNN(NeuralNetwork):
    def __init__(self):
        super().__init__()
        self.set_hidden(self.hidden_layers())
        self.set_cls(self.classifier())
        self.set_params(self._cls.trainable_variables + self._hidden.trainable_variables)

    def __repr__(self):
        return print("<Fully Connencted NeuralNetwork>")

    def hidden_layers(self, input_shape = 28*28, neurons = 50):
        hidden = Sequential([layers.InputLayer(input_shape=input_shape)
                , layers.Dense(neurons)
                , layers.Dense(neurons)])
        self._last_layer_neurons = neurons
        return hidden