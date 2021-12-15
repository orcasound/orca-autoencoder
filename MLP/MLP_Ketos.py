import tensorflow.keras as tk
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model
from ketos.data_handling.data_feeding import BatchGenerator
from ketos.data_handling.database_interface import table_description
from ketos.data_handling.database_interface import table_description_annot
import tensorflow.keras as tfk
from ketos.audio.spectrogram import Spectrogram, MagSpectrogram
from ketos.neural_networks.dev_utils.nn_interface import NNInterface
import pandas as pd
import ketos.data_handling.database_interface as dbi
import ketos
from random import random
import os
from ketos.neural_networks.dev_utils import  nn_interface   # RecipeCompat,
##################################
#################################
outputDir = "dbs/"            # this is a directory for the checkpoints
os.mkdir(outputDir)
########################  h5 database
def buildH5database(X_train, Y_train, X_test, Y_test, testFraction):
    h5filename = 'mnist.h5'
    h5file = dbi.open_file(outputDir + h5filename, 'w')
#    for X in X_train:
    for i in range(100):
        spec = MagSpectrogram.empty()
        spec.data = X_train[i]     # use empty spectrogram to put numpy array (X) and label in a form that ketos likes
        spec.label = Y_train[i]
        frac = random()
        if frac < testFraction:
            descr_data = table_description(spec,  include_label=True, include_source=False)
            tbl_data = dbi.create_table(h5file, "/train/", "table_data", descr_data)
            dbi.write(spec, tbl_data)
            tbl_data.flush()
        else:
            descr_data = table_description(spec, include_label=True, include_source=False)
            tbl_data = dbi.create_table(h5file, "/test/", "table_data", descr_data)
            dbi.write(spec, tbl_data)
            tbl_data.flush()
    return h5file

########################
class MLP(Model):
    def __init__(self, n_neurons, activation):
        super(MLP, self).__init__()
        self.dense = tfk.layers.Dense(n_neurons, activation=activation)
        self.final_node = tfk.layers.Dense(10)

    def call(self, inputs):
        output = self.dense(inputs)
        output = self.dense(output)
        output = self.final_node(output)
        return output
    
class MLPInterface(NNInterface):
    def __init__(self, n_neurons, activation, optimizer, loss_function, metrics):
        super(MLPInterface, self).__init__(optimizer, loss_function, metrics)
        self.n_neurons = n_neurons
        self.activation = activation
        self.model = MLP(n_neurons=n_neurons, activation=activation)



#######################
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()   # load the mnist train and test data

testFraction = 0.10
h5_db = buildH5database(X_train, Y_train, X_test, Y_test, testFraction)  # I used a MagSpectrogram.empty() object here
                                                                         #  to get a place to put the label with the mnist array
train_data = dbi.open_table(h5_db, "/train/table_data")
test_data = dbi.open_table(h5_db, "/test/table_data")

train_generator = BatchGenerator(data_table=train_data, annot_in_data_table=True, batch_size=32, x_field='data', return_batch_ids=False)
val_generator =  BatchGenerator(data_table=test_data,   annot_in_data_table=True, batch_size=32, x_field='data', return_batch_ids=False)

# y = next(val_generator)  # y[0] is numpy array ( num specs, ndimx, ndimy)
                           # y[0][0] is first (ndimx, ndimy) numpy array
                           # y[1][0] is first (integer) label

#  I found these metrics somewhere in ketos so tossed them in here  --
# Example Precision
p = tk.metrics.Precision
dec_p = ketos.neural_networks.dev_utils.nn_interface.RecipeCompat("precision", p)

 # Example Optimizer
opt = tk.optimizers.Adam
dec_opt = ketos.neural_networks.dev_utils.nn_interface.RecipeCompat("adam", opt, learning_rate=0.001)

 # Example Loss
loss = tk.losses.BinaryCrossentropy
dec_loss = ketos.neural_networks.dev_utils.nn_interface.RecipeCompat('binary_crossentropy', loss, from_logits=True)

n_neurons = 784
activation = 'relu'
optimizer = 'Adam'
loss_function = 'CrossEntropy'
metrics = (dec_p, dec_loss, dec_opt)  # maybe those metrics go in like this???
mnist_classifier = MLPInterface(n_neurons, activation, optimizer, loss_function, metrics)
mnist_classifier.train_generator = train_generator    # I have read that I should flatten the 2D arrays, but how and where??
mnist_classifier.val_generator   = val_generator
mnist_classifier.checkpoint_dir = outputDir + "checkpoints"

mnist_classifier.train_loop(n_epochs=30, verbose=True)  # this throws an error:
                                    # /nn_interface.py", line 1234, in train_loop
                                    #     train_metric.reset_states()
                                    # AttributeError: 'BinaryCrossentropy' object has no attribute 'reset_states'


