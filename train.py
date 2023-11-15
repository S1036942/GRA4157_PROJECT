import argparse

from neuralnetwork import NeuralNetwork
from fullyConNN import FullyConNN
from convNN import ConvNN

from dataloader import DataLoader
from cifar import CIFAR10
from mnist import MNIST

def train_test():
    parser = argparse.ArgumentParser(prog = "Animal test program", 
                                    formatter_class = argparse.RawDescriptionHelpFormatter, 
                                    description = textwrap.dedent('''\
                                                NeuralNetwork
                                    --------------------------------
                                    A program that creates animals and greets for each animal
                                    
                                    Methods:
                                    1) getName:
                                    @return name the name of the Animal
                                    
                                    2) 
                                    Each type of animal has its own way of greeting.
                                    '''),
                                    epilog=textwrap.dedent('''\
                                                Usage
                                    --------------------------------
                                    # Initializes objects of different animals
                                    e = Animal("Esteban")           
                                    b = Dog("Bence")
                                    k = BigDog("Karl")
                                    c = Cat("Cleo")

                                    print(e.getName())
                                    print("Expected: Esteban")

                                    b.greet()
                                    print("Expected: Woof")

                                    k.greet()
                                    print("Expected Woof\nWoooof")

                                    c.greet()
                                    print("Expected: Meow")

                                    '''))
    parser.add_argument('--run_test', action='store_true', help='runs NeuralNetwork test')
    #parser.add_argument('--nn_type', action='store_true', default= FullyConNN, help='Specifies what kind of Neuralnetwork to use.')
    type = parser.add_argument('--nn_type', type = str, choices = ["FullyConNN", "ConvNN"], help='Specifies what kind of Neuralnetwork to use.')
    epochs = parser.add_argument('--epochs', type = int, default = 10, help='Specifies the number of epochs.')
    neurons = parser.add_argument('--neurons', type = int, default = 50, help='Specifies the number of neurons.')
    batch_size = parser.add_argument('--batch_size', type = int, default = 256, help='Specifies the size of the batch size.')
    dset = parser.add_argument('--dset', type = str, choices = ["MNIST", "CIFAR10"], help='Specifies which data set to use')


    args = parser.parse_args()
    if args.run_test:
        try:
            assert isinstance(type, str)
            assert isinstance(epochs, int)
            assert isinstance(epochs, int)

            if type == "FullyCoNN":
                type = FullyConNN()
            elif type == "ConvNN":
                pass


        
        except AssertionError as e:
            print(e)


