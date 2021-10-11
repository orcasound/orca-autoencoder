
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import ae_classes_1
import numpy as np
import pickle
import time
import math
from random import random
import os
import helpers_3D as d3

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import ReLU, BatchNormalization, Reshape, Flatten

from tensorflow import keras

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print('device is', device)


def saveModel(model, saveFilename):
    with open(saveFilename, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def loadModel(loadFilename):
    with open(loadFilename, 'rb') as f:
        return pickle.load(f)

def encoderCNN(inputs, layers):
    """ Construct the Encoder
        inputs : the input vector
        layers : number of filters per layer    """
    x = inputs
    # Feature pooling by 1/2H x 1/2W
    for n_filters in layers:
        x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    return x


def decoderCNN(x, layers):
    """ Construct the Decoder
      x      : input to decoder
      layers : the number of filters per layer (in encoder)
    """
    # Feature unpooling by 2H x 2W
    for _ in range(len(layers) - 1, 0, -1):
        n_filters = layers[_]
        x = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False,
                            kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    # Last unpooling, restore number of channels
    x = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
########################################
def loadSpectra(specPklDir, validFrac):
    fileList = []
    itms = os.listdir(specPklDir)
    for it in itms:
        if 'pkl' in it and 'Summary' not in it:
            fileList.append(specPklDir + it)
    dblist = []
    lbls = []
    iStart = 0
    for pklfile in fileList:
        print('Reading file ', pklfile)
        specObjs = d3.load_obj(pklfile)

        for ary in specObjs.arrayList:
            if len(ary) > 0:  ######  Note Bene  Only if averages have > zero length
                for ar in ary:
                    dblist.append(ar)
    specs_train = []
    specs_valid = []
    for i in range(len(dblist)):
        if True not in np.isnan(dblist[i].tolist()):  # DO NOT ACCEPT SPECTROGRAMS THAT HAVE nans
            if random() < validFrac:
                specs_valid.append(dblist[i])
            else:
                specs_train.append(dblist[i])
        else:
            print("GOT NANS in array number ", i)
    return (specs_train, specs_valid)

#####################################################  RUN RUN RUN
thisdate = '10_08'
modelID = 'CNN_1'
init_epochs = 1000
addnl_epochs = 10

loadModelFilename = "../../models/CNN_1"   # model to use at START of new run
saveModelDir = "../../models/CNN_1" ##model_{}_{}_{}.pkl".format(modelID, init_epochs + addnl_epochs, thisdate)
specPklDir = "../../spectrograms_64_64/"
jpgDir = "../../jpgs/"

# Parameters
params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 6}
validation_fraction = 0.1
# Datasets
#training_generator, validation_generator, rows, cols = ae_classes_1.setupOrcaDatasets(device, params,                                                                                     validation_fraction, specPklDir)
# Generators are in my_classes.py
#print("generators have ", rows, " and ", cols, "cols")

# if loadModelFilename != "":
#     model = loadModel(loadModelFilename)
#     ##    model.cuda()
#     print('************  this existing model ************************')
#     print(model)
#     print("     model device is ", next(model.parameters()).device)
#     print('     try model.eval')
#     model.eval()
#     print('     try doValidation')
#     doValidation(jpgDir + 'model_{}_at_epoch_{}.jpg'.format(modelID, init_epochs))
#
# else:
#     model = ae_classes_1.MLP_4(input_shape=rows * cols).to(device)
#     print("new model is a MLP:", model)

###########################
(x_train, x_test) = loadSpectra(specPklDir,validation_fraction)
print('lengths', len(x_train), len(x_test))
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# metaparameter: number of filters per layer in encoder
layers = [64, 32, 32]
# The input tensor
inputs = Input(shape=(64, 64, 1))
# The encoder
x = encoderCNN(inputs, layers)
# The decoder
outputs = decoderCNN(x, layers)

# Instantiate the Model
if loadModelFilename != "":
    aeCNN = keras.models.load_model(loadModelFilename)
    score = aeCNN.evaluate(x_test, x_test)
    print('Starting with Test loss:', score[0])
    print('Starting with Test accuracy:', score[1])
else:
    aeCNN = Model(inputs, outputs)
print(aeCNN.summary())


tstart = time.time()
aeCNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
aeCNN.fit(x_train, x_train, epochs=addnl_epochs, batch_size=32, validation_split=validation_fraction, verbose=1)
print('call model evaluate(x_test, x_test')
score = aeCNN.evaluate(x_test, x_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
####aeCNN.save(saveModelDir)
tstop = time.time()
print("Elapsed time s {}, m {}, hr {:.2f} s/epoch {:.2f} ".format(int(tstop - tstart), int((tstop - tstart) / 60.0), ((tstop - tstart) / 3600.0),
    (tstop - tstart) / addnl_epochs))
##########################

