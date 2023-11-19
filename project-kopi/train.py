import argparse
import textwrap
import sys
import tensorflow as tf

from neuralnetwork import NeuralNetwork
from fullyConNN import FullyConNN
from convNN import ConvNN

from dataloader import DataLoader
from cifar import CIFAR10
from mnist import MNIST

def train_test():
    classes = {"neuralnetwork" : NeuralNetwork, "fully_con" : FullyConNN, "conv" : ConvNN, "dataloader" : DataLoader, "cifar10" : CIFAR10, "mnist" : MNIST}
    parser = argparse.ArgumentParser(prog = "NeuralNetwork test program", 
                                    formatter_class = argparse.RawDescriptionHelpFormatter, 
                                    description = textwrap.dedent('''\
                                    Classes used in this program:

                                                NeuralNetwork
                                    --------------------------------
                                                FullyConNN
                                    --------------------------------
                                                ConvNN
                                    --------------------------------
                                                DataLoader
                                    --------------------------------
                                                MNIST
                                    --------------------------------
                                                CIFAR10
                                    --------------------------------

                                    To see the public interface of each class, pass inn argument:
                                    -- doc <classname> (See list of them in options)
                                 
                                    '''),
                                    epilog=textwrap.dedent('''\
                                                Usage
                                    --------------------------------
                                    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)    # Creates an optimizer
                                    #tr_data = (dset.x_tr, dset.y_tr)                           # Get the training X and Y values
                                    tr_data = dset.loader(batch_size)                           # Create loader object

                                    step =0
                                    while step < args.epochs:
                                        for i, data_batch in enumerate(tr_data):            
                                            losses = model.train(data_batch, optimizer)         # Train model
                                        step +=1

                                    y_and_pi = model.test(dset.x_te)                            # Calculate y_hat and pi_hat 
                                    y_te = y_and_pi[0]
                                    pi_hat = y_and_pi[1]
                                    data_loader = dset
                                    
                                    # Calculate AUC score
                                    from sklearn.metrics import roc_auc_score 
                                    auc = roc_auc_score(data_loader.y_te, pi_hat, multi_class='ovr')
                                    #auc = roc_auc_score(data_loader.y_te, pi_hat, multi_class='ovo')
                                    print ('final auc %0.4f' % (auc))
                                    

                                    '''))
    parser.add_argument('--nn_type', type = str, choices = ["fully_con", "conv"], help='Specifies the Neuralnetwork.')
    parser.add_argument('--epochs', type = int, default = 10, help='Specifies the number of epochs.')
    parser.add_argument('--neurons', type = int, default = 50, help='Specifies the number of neurons.')
    parser.add_argument('--batch_size', type = int, default = 256, help='Specifies the size of the batch size.')
    parser.add_argument('--dset', type = str, choices = ["mnist", "cifar10"], help='Specifies which data set to use')
    parser.add_argument('--doc', type = str, choices = classes.keys(), help='Choose the doctring you want to see: --doc <name>')


    args = parser.parse_args()  

    # If it is only doc string and no other input:
    if args.doc and not any([args.nn_type, args.epochs, args.neurons, args.batch_size, args.dset]):
        the_class = classes[args.doc]
        print(the_class.__doc__)
    else: 
        model = args.nn_type
        epochs = args.epochs
        neurons = args.neurons
        batch_size = args.batch_size
        dset = args.dset
        doc = args.doc
        try:

            #Print the doc string if it is specified together with others:
            if doc:
                the_class = classes[args.doc]
                print(the_class.__doc__)

            """
            assert isinstance(model, str)
            assert isinstance(epochs, int)
            assert isinstance(neurons, int)
            assert isinstance(batch_size, int)
            assert isinstance(dset, str)"""
            
            if model is not None:
                assert isinstance(model, str)
            if epochs is not None:
                assert isinstance(epochs, int)
            if neurons is not None:
                assert isinstance(neurons, int)
            if batch_size is not None:
                assert isinstance(batch_size, int)
            if dset is not None:
                assert isinstance(dset, str)
            
            # Check The Type of neural network:
            if model:
                if model == "fully_con":
                    model = FullyConNN()
                elif model == "conv":
                    model = ConvNN()
                else:
                    print("You did not specify the Neural Network class correctly")
                    sys.exit()
                
                # Check The data set
                if dset == "mnist":
                    dset = MNIST()
                elif dset == "cifar10":
                    dset = CIFAR10()
                else:
                    print("You did not specify the DataLoader class correctly")
                    sys.exit()

                # Check that the model works with the data set
                if (model == "fully_con" and dset == "cifar10"):
                    print("Can not run FullyConNN on the CIFAR10 data.\nThe model is not made for this picture format")
                    sys.exit()
                elif (model == "conv" and dset == "mnist"):
                    print("Can not run ConvNN on the MNIST data.\nThe model is not made for this picture format.")
                    sys.exit()
                else:
                    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
                    #tr_data = (dset.x_tr, dset.y_tr)
                    tr_data = dset.loader(batch_size)

                    step =0
                    while step < args.epochs:
                        for i, data_batch in enumerate(tr_data):
                            losses = model.train(data_batch, optimizer)
                        step +=1

                    y_and_pi = model.test(dset.x_te)
                    y_te = y_and_pi[0]
                    pi_hat = y_and_pi[1]
                    data_loader = dset

                    from sklearn.metrics import roc_auc_score 
                    auc = roc_auc_score(data_loader.y_te, pi_hat, multi_class='ovr')
                    #auc = roc_auc_score(data_loader.y_te, pi_hat, multi_class='ovo')
                    print ('final auc %0.4f' % (auc))               

        except AssertionError as e:
            print(e)
            print("Assertion error. One of the input values was not correct format")
            sys.exit()

train_test()

