import numpy
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
from scipy import signal

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import ReLU, BatchNormalization, Reshape, Flatten

from tensorflow import keras

########################################################################
def savePkl(saveObj, saveFilename):
    with open(saveFilename, 'wb') as f:
        pickle.dump(saveObj, f, pickle.HIGHEST_PROTOCOL)


def loadPkl(loadFilename):
    with open(loadFilename, 'rb') as f:
        return pickle.load(f)

def loadAndSaveSpectra(specPklDir, validFrac):
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
    result = (specs_train, specs_valid)
    savePkl(result, specPklDir + 'specArrays_{}_.pkl'.format(validFrac))
    return result

def loadSpectra(specPklDir):
    result = loadPkl(specPklDir)  # returns (specs_train, specs_valid)
    return result

def plotBeforeAndAfter(befores, afters, ncols, NtoPlot, offset):
    nrows = NtoPlot // ncols  # number of rows of PAIRS
    plt.figure(figsize=(25, 25 * (nrows * 2) // ncols))
    index = 0
    for j in range(nrows):
        for i in range(ncols):
            ax = plt.subplot(nrows * 3, ncols, index + i + 1 + j * 2 * ncols)
            plt.title('Input number {}'.format(offset+index + i))
            plt.imshow(befores[offset+index + i])
#            plt.imshow(test_examples[index + i].detach().cpu().numpy().reshape(rows, cols) + 0.01)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        for i in range(ncols):
            diff = np.subtract(np.asarray(np.squeeze(befores[offset+index + i])), np.asarray(np.squeeze(afters[offset+index + i])))
            manhattenNorm = np.sum(np.abs(diff))
            diff2 = np.subtract(np.asarray(np.squeeze(befores[offset+index + i])), np.asarray(np.squeeze(afters[offset+index + i +1])))
            manhattenNorm2 = np.sum(np.abs(diff2))

            ax = plt.subplot(nrows * 3, ncols, index + ncols + i + 1 + j * 2 * ncols)
            plt.title('diff is {:0.0f}, adjacent is {:0.0f}'.format(manhattenNorm, manhattenNorm2))
            plt.imshow(afters[offset+index + i])
            #plt.imshow(diff)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        for i in range(ncols):
            diff = np.subtract(np.asarray(np.squeeze(befores[offset+index + i])), np.asarray(np.squeeze(afters[offset+index + i])))
            ax = plt.subplot(nrows * 3, ncols, index + 2 * ncols + i + 1 + j * 2 * ncols)
            plt.title('Difference between input and reconstruction')
            plt.imshow(diff)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        index += ncols
    plt.show()
    #plt.savefig(jpgFilename)


#########################################################################
loadModelFilename = "../../models/CNN_1"
jpgDir = "../../jpgs/"
specPklDir = "../../spectrograms_64_64/"
specArrayFile = '/preparedArrays/specArrays_0.1_.pkl'
validation_fraction = 0.1

aeCNN = keras.models.load_model(loadModelFilename)
print(aeCNN.summary())
###########################
if specArrayFile == "":
    (x_train, x_test) = loadAndSaveSpectra(specPklDir, validation_fraction)   # only need to run once to build and save
else:
    (x_train, x_test) = loadSpectra(specPklDir + specArrayFile)
print('spec shape is ', x_test[261].shape)
print('x_train has length', len(x_train), 'x_test has length', len(x_test))
# score = aeCNN.evaluate(x_test, x_test, verbose = 1)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

x_fordisplay = np.expand_dims(x_test, axis=-1)

#[261] is a nice spectrgram

x_predict = aeCNN.predict(x_fordisplay)
rows = cols = 64
x_image = x_predict[261]

offset = 250
numSpecs = 8
ncols = 4
for i in range(10):
    if offset + i * numSpecs < len(x_predict):
        plotBeforeAndAfter(x_test, x_predict, ncols, numSpecs, offset + i * numSpecs)


