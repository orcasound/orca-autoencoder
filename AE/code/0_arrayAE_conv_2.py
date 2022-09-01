#!/usr/bin/python3
"""
to run a py file:
cd PyCharm/TF_val
tensorman =MY_IMAGE run --gpu python3 code/arrayAE_conv_2.py


to just run python3 etc:
  tensorman run --gpu --python3 --root --name MY_CONTAINER bash
      install needed packages via  python -m pip install xxxxx
      then exit

help from:
    https://chat.pop-os.org/pop-os/messages/@mmstick      
"""

import sys
import helpers
import importlib
importlib.reload(helpers)



import numpy as np

import matplotlib.pyplot as plt
import os
import math
import time
import random 


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import ReLU, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers

import pickle
from tensorflow.keras.models import load_model
import h5py
###################################################

        

def train_autoencoder(n_neurons, X_train, X_valid, loss, optimizer,
                      n_epochs=10, output_activation=None, metrics=None):
    n_inputs = X_train.shape[-1]
    encoder = keras.models.Sequential([
        keras.layers.Dense(n_neurons, activation="selu", input_shape=[n_inputs])
    ])
    decoder = keras.models.Sequential([
        keras.layers.Dense(n_inputs, activation=output_activation),
    ])
    autoencoder = keras.models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer, loss, metrics=metrics)
    autoencoder.fit(X_train, X_train, epochs=n_epochs,
                    validation_data=(X_valid, X_valid))
    return encoder, decoder, encoder(X_train), encoder(X_valid) 
    

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))
                                
#####################################################
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

###########################################################

def generate_arrays_from_h5(h5file, group, group_label, batchsize):
    inputs = []
    targets = []
    batchcount = 0
    lineCnt = 0
    print("top of generate")
    while True:
        specs = h5file[group][batchcount:batchcount+batchsize].data
        labels = h5file[group_label][batchcount:batchcount+batchsize].data
        batchcount += 1
        if batchcount % 10 == 0:
            print("batchcount=", batchcount, "batchsize=", batchsize)
        yield (specs.obj, labels.obj)  #  Note Bene - for auto encoder target is input!

def generate_AE_arrays_from_h5PRIOR(h5file, AE, group, group_label, batchsize):  # AE is T/F for AutoEncoder style yield
    inputs = []
    targets = []
    batchcount = 0
    lineCnt = 0
    print("top of generate")

    while True:
        specs = h5file[group][batchcount:batchcount+batchsize]
        specs = np.expand_dims(specs, axis=-1)   # convert specs to (100, 256, 256, 1)
        if group_label != None:
            labels = h5file[group_label][batchcount:batchcount+batchsize].data

        batchcount += 1
        if batchcount % 10 == 0:
            print("batchcount=", batchcount, "batchsize=", batchsize)
        if AE:
            yield(specs, specs)  #  Note Bene - for auto encoder target is input!
        else:
            yield(specs, labels) 
            
def generate_AE_arrays_from_h5(h5file, AE, group, group_label, batchsize):  # AE is T/F for AutoEncoder style yield
    inputs = []
    targets = []
    batchcount = 0
    lineCnt = 0
    print("\n----------------------------top of generate", batchcount, batchsize)
#    print("h5file keys", group, h5file.keys())
    while True:

        specs = h5file[group][batchcount:batchcount+batchsize]#[0] #.data
        specs = np.expand_dims(specs, axis=-1)   # convert specs to (100, 256, 256, 1)
        #print("-------------------", specs.shape)
        if group_label != None:
            labels = h5file[group_label][batchcount:batchcount+batchsize].data

        batchcount += 1
        if batchcount % 10 == 0:
            print("batchcount=", batchcount, "batchsize=", batchsize)
        if AE:
            yield(specs, specs)  #  Note Bene - for auto encoder target is input!
        else:
            yield(specs, labels)    

def getSignal(noise, Ntimes, Npsds):
    # create some horizontal lines as 'signal
    Nlines = 4
    for i in range(Nlines):
        ht = random.randint(5, Npsds - 5)
        x  = random.randint(5, Ntimes//2)
        for j in range(Ntimes//3):
            noise[ht, x+j] = 1
            noise[ht-1, x + j] = 1
            noise[ht+1, x + j] = 1
    return noise

def buildFakeSpectra(Nrecords, Ntimes, Npsds):
    h5filename = "h5fakeSpecsSml.h5"
    h5db = h5py.File("data/" + h5filename, mode='w')
    trainSpecs = []
    trainLbls = []
    testSpecs = []
    testLbls = []
    tests = []
    for i in range(Nrecords):
        if i % 100 == 0:
            print(i, "of", Nrecords)
        specNoise = np.random.rand(Ntimes, Npsds)
        specSignal = getSignal(np.random.rand(Ntimes, Npsds), Ntimes, Npsds)
        if random.random() < 0.75:
            trainSpecs.append(specSignal)
            trainLbls.append(1)
            trainSpecs.append(specNoise)
            trainLbls.append(0)
        else:
            testSpecs.append(specSignal)
            testLbls.append(1)
            testSpecs.append(specNoise)
            testLbls.append(0)

    temp = list(zip(testSpecs, testLbls))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    testSpecs, testLbls = list(res1), list(res2)
    print("Randomize the records")
    temp = list(zip(trainSpecs, trainLbls))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    trainSpecs, trainLbls = list(res1), list(res2)
    print("fill the h5 database")
    h5db['train_specs'] = trainSpecs
    h5db["train_labels"] = trainLbls
    h5db['test_specs'] = testSpecs
    h5db["test_labels"] = testLbls
    print("some labels", h5db["train_labels"])
#    print("Plot some examples of fake data")
#    for i in range(10):
#        plt.imshow(h5db['train_specs'][i])
#        plt.title("Label is {}".format(h5db["train_labels"][i]))
#        plt.show()
    h5db.close()
    return h5filename

##########################################################

IMAGES_PATH = helpers.makeDir("../outputFiles/AEplots/")
INPUT = 1
NN = 6
prior_epochs = 2000
nEpochs = 2000
batchsize = 50
useFakeData = False
newModelID = "09_12_21_passby_6"

chkptFile = helpers.makeDir("../models/" + "{}_best.ckpt".format(newModelID))

#h5filename = "data/" + "h5fakeSpecsSmlLabeled.h5"
#useFakeData = True
#newModelID = "fakedata_0"


h5filename2 = "db/OS_09_12_21SpecsNormedSquaredSml.h5"  # this db leads off with some orca calls
h5filename2 = "../h5files/OS_09_12_21SpecsNormedSquared_11_wavs.h5"
h5filename = "../h5files/OS_09_12_21SpecsNormedSquared_11_wavs.h5"



total_epochs = prior_epochs + nEpochs
### setup the data input
# initialize generators
if useFakeData and h5filename == "":
    h5filename = buildFakeSpectra(1000, 256, 256)  # build h5 database with this many records of shape (1000, 256, 256)

h5file = h5py.File(h5filename, 'r')
h5file2 = h5py.File(h5filename2, 'r')

print(h5filename, "keys are", h5file.keys())
for key in h5file.keys():
    print(key, "length", len(h5file[key]))

if useFakeData:
    train = generate_AE_arrays_from_h5PRIOR(h5file, True, 'train_specs', 'train_labels', batchsize)  # train is (spectrograms, labels)
    test = generate_AE_arrays_from_h5PRIOR(h5file, True, 'test_specs', 'test_labels', batchsize)
    #val = generate_AE_arrays_from_h5PRIOR(h5file, True, 'eval_specs', 'eval_labels', batchsize)
else:
    train = generate_AE_arrays_from_h5(h5file, True, 'train_specs', None, batchsize)
    test  = generate_AE_arrays_from_h5(h5file, True, 'test_specs', None, batchsize)
    evalAry = generate_AE_arrays_from_h5(h5file2, True, 'eval_specs', None, batchsize)
    print("----------------")
    #evalspecs = next(evalAry)  

chkptFile = "models/" + "{}_best.ckpt".format(newModelID)
checkpoint = ModelCheckpoint(chkptFile, monitor='loss', verbose=1,
                             save_best_only=True, mode='auto', period=1)
tstart = time.time()

#########################################################################


class AE_0(Model):
  def __init__(self):
    super(AE_0, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(ndim, ndim, 1)),
#      layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Input(shape=(32, 32, 8)),
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
#      layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

if NN == 4:   # simplest CNN autoencoder via Keras Sequential API

    X_train = np.expand_dims(X_train, axis=-1)
    X_valid = np.expand_dims(X_valid, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    conv_ae = AE_0()  
    print("\nNN =", NN, "ndim=", ndim, "nEpochs=", nEpochs)
    print("\nEncoder:")
    print(conv_ae.encoder.summary())
    print("conv_ae.encoder.output_shape", conv_ae.encoder.output_shape)
    print("\nDecoder:")
    print(conv_ae.decoder.summary())
    print("conv_ae.decoder.output_shape", conv_ae.decoder.output_shape)
    conv_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=1.0),\
     metrics=[rounded_accuracy])
    history = conv_ae.fit(X_train, X_train, epochs=nEpochs,
                      validation_data=(X_valid, X_valid))
    helpers.show_reconstructions(conv_ae, X_valid, IMAGES_PATH, "conv_ae_32_16_8_ndim_{}_epochs_{}_NN_{}".format(ndim, nEpochs, NN))
    
    print('call model evaluate(X_test, X_test')
    score = conv_ae.evaluate(X_test, X_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])     
    helpers.show_history(history, len(X_test), score, IMAGES_PATH, "conv_aeHistory_32_16_8_ndim_{}_epochs_{}_NN_{}".format(ndim, nEpochs, NN))
    print(conv_ae.summary())
    
    
    # save trained networks
    filename = "data/" + "conv_ae_obj_N_{}_epochs_{}".format(N, nEpochs)
    print("saving", filename)
    conv_ae.save(filename)   #     Encoder _ bottlenect size
    filename = "data/" + "conv_ae_Encoder_32_32_obj_N_{}_epochs_{}".format(N, nEpochs)
    print("saving", filename)
    conv_ae.encoder.save(filename)

if NN == 5:
    modelFilename = "conv_ae_09_12_21_passby_sml_0_NN_5_epochs_60_10"#"conv_ae_obj_NN_5_epochs_50_50"  #"conv_ae_obj_N_4000_epochs_50"
    modelFilenameWts = modelFilename + "fullNNwts"
    print("loading....", modelFilename)
    ndim = 256
    conv_ae = AE_0() #load_model("data/" + modelFilename)  
    conv_ae.load_weights("data/" + modelFilenameWts)
    print("\nNN =", NN, "ndim=", ndim, "# of nEpochs to fit=", nEpochs)
    print("\nEncoder:")
    print(conv_ae.encoder.summary())
    print("conv_ae.encoder.output_shape", conv_ae.encoder.output_shape)
    print("\nDecoder:")
    print(conv_ae.decoder.summary())
    print("conv_ae.decoder.output_shape", conv_ae.decoder.output_shape)
    conv_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=1.0),\
     metrics=[rounded_accuracy])
    print("\nprior to fit newModelID", newModelID) 
    evalspecs = next(evalAry)  
    
    helpers.show_reconstructionsSept(conv_ae, evalspecs[0], "plots/", "conv_ae_{}_reconstruction_epochs_{}priorFit".format(newModelID, prior_epochs), 10)

#    print("Confusion matrix before running fit ONLY FOR LABELED DATA")
#    helpers.printConfusionMatrix(conv_ae, "train dataset", next(train)) 
    
    history = conv_ae.fit(train, steps_per_epoch=20, initial_epoch=prior_epochs,epochs=total_epochs, verbose=2, validation_data=test, validation_steps=20, callbacks=[checkpoint])
      
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss {}'.format(format(newModelID)))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("plots/" + "Epochs_{}_{}_model_{}_loss.jpg".format(prior_epochs, nEpochs, newModelID))
    plt.close()  
    # save trained networks
    filename = "data/" + "conv_ae_{}_NN_{}_epochs_{}_{}".format(newModelID, NN, prior_epochs, nEpochs)
    print("\nsaving", filename)
    conv_ae.save_weights(filename+"fullNNwts")
    print("weights saved!")
#    evalspecs = next(test)  
#    print("evalspecs[0][0]", evalspecs[0][0])  
    helpers.show_reconstructionsSept(conv_ae, evalspecs[0], "plots/", "conv_ae_{}_reconstruction_at_epochs_{}postFit".format(newModelID, prior_epochs + nEpochs), 10)


class AE_6(Model):
  def __init__(self):
    super(AE_6, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(ndim, ndim, 1)),
      layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Input(shape=(16, 16, 8)),
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
if NN == 6:
    modelFilename = "../models/conv_ae_09_12_21_passby_6_NN_6_epochs_2000_2000fullNNwts"
#    modelFilename = "/home/val/PyCharmFiles/TF_val/data/conv_ae_09_12_21_passby_6_NN_6_epochs_2000_2000fullNNwts"
#    modelFilename = "/home/val/PyCharmFiles/TF_val/data/conv_ae_09_12_21_passby_6_NN_6_epochs_1000_1000fullNNwts"
    print("loading....", modelFilename)
    ndim = 256
    conv_ae = AE_6() #load_model("data/" + modelFilename)  
    conv_ae.load_weights(modelFilename)
    conv_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=0.001),\
     metrics=[rounded_accuracy])
    evalspecs = next(evalAry) 

    x = evalspecs[0] 
    print("x shape", x.shape)
    score = conv_ae.evaluate(x,x,verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) 

    print("\nNN =", NN, "ndim=", ndim, "# of nEpochs to fit=", nEpochs)
    print("\nEncoder:")
    print(conv_ae.encoder.summary())
    print("conv_ae.encoder.output_shape", conv_ae.encoder.output_shape)
    print("\nDecoder:")
    print(conv_ae.decoder.summary())
    print("conv_ae.decoder.output_shape", conv_ae.decoder.output_shape) 
    print("\nprior to fit newModelID", newModelID) 
        
    helpers.show_reconstructionsSept(conv_ae, evalspecs[0], "plots/", "conv_ae_{}_reconstruction_epochs_{}priorFit".format(newModelID, prior_epochs), 10)
    
    history = conv_ae.fit(train, steps_per_epoch=20, initial_epoch=prior_epochs,epochs=total_epochs, verbose=2, validation_data=test, validation_steps=20, callbacks=[checkpoint])
    plt.close("all")  
    print("history keys", history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss {}'.format(format(newModelID)))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("plots/" + "Epochs_{}_{}_model_{}_loss.jpg".format(prior_epochs, nEpochs, newModelID))
    plt.close()  
    # save trained networks
    filename = "data/" + "conv_ae_{}_NN_{}_epochs_{}_{}".format(newModelID, NN, prior_epochs, nEpochs)
    print("\nsaving", filename)
    conv_ae.save_weights(filename+"fullNNwts")
    print("weights saved!")
#    evalspecs = next(test)  
#    print("evalspecs[0][0]", evalspecs[0][0])  
    helpers.show_reconstructionsSept(conv_ae, evalspecs[0], "plots/", "conv_ae_{}_reconstruction_at_epochs_{}postFit".format(newModelID, prior_epochs + nEpochs), 10)
    print('call model evaluate evalspecs')
    score = conv_ae.evaluate(evalspecs[0], evalspecs[0])
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) 
    helpers.show_history(history, len(evalspecs[0]), score, IMAGES_PATH, "conv_ae_{}_History_32_16_8_ndim_{}_epochs_{}_NN_{}".format(newModelID, ndim, nEpochs, NN))
    print(conv_ae.summary())
    
    
        


##################################################################
tstop = time.time()
print("Elapsed time s {}, m {}, hr {:.2f} s/epoch {:.2f} ".format(int(tstop - tstart),\
 int((tstop - tstart) / 60.0),((tstop - tstart) / 3600.0),(tstop - tstart) / nEpochs))    
    
