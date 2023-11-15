from neuralnetwork import NeuralNetwork
from fullyConNN import FullyConNN
from convNN import ConvNN

"""
nn = NeuralNetwork()
nn.__repr__()

fn = FullyConNN()
fn.__repr__()

cn = ConvNN()
cn.__repr__()

"""

from dataloader import DataLoader
from cifar import CIFAR10
from mnist import MNIST

dl = DataLoader()

cifar = CIFAR10()
print(cifar.x_tr)

mnist = MNIST()