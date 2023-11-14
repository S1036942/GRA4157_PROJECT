from neuralnetwork import NeuralNetwork
from  tensorflow.keras import  layers
from  tensorflow.keras.models  import  Sequential

class ConvNN(NeuralNetwork):
    def __init__(self):
        super().__init__()
        self.set_hidden(self.hidden_layers())
        self.set_cls(self.classifier())
        self.set_params(self._cls.trainable_variables + self._hidden.trainable_variables)

    def __repr__(self):
        print("<Convolutional NeuralNetwork>")


    def hidden_layers(self, input_shape = (32,32,3), neurons = 50, filters = 32, kernel_size = 3, strides = (2,2)):
        hidden = Sequential([layers.InputLayer(input_shape=input_shape)
                ,layers.Conv2D(filters=filters ,kernel_size=kernel_size ,strides=strides)
                ,layers.Conv2D(filters =2* filters ,kernel_size=kernel_size ,strides=strides)
                ,layers.Conv2D(filters=neurons ,kernel_size=kernel_size ,strides =(5 ,5))
                ,layers.Flatten ()])
        self._last_layer_neurons = neurons
        return hidden
    
    
        

