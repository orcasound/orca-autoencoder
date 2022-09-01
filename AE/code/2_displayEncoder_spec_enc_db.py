#!/usr/bin/python3
"""

"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import math
import h5py
import helpers as d3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


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


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


###########################################################

def generate_arrays_from_h5(h5file, group, group_label, batchsize):
    inputs = []
    targets = []
    batchcount = 0
    lineCnt = 0
    print("top of generate")
    while True:
        specs = h5file[group][batchcount:batchcount + batchsize].data
        labels = h5file[group_label][batchcount:batchcount + batchsize].data
        batchcount += 1
        if batchcount % 10 == 0:
            print("batchcount=", batchcount, "batchsize=", batchsize)
        yield (specs.obj, labels.obj)  # Note Bene - for auto encoder target is input!


def generate_AE_arrays_from_h5(h5file, AE, group, group_label, batchsize):  # AE is T/F for AutoEncoder style yield
    inputs = []
    targets = []
    batchcount = 0
    lineCnt = 0
    print("\n----------------------------top of generate", batchcount, batchsize)
    #    print("h5file keys", group, h5file.keys())
    while True:
        specs = h5file[group][batchcount:batchcount + batchsize]  # [0] #.data
        specs = np.expand_dims(specs, axis=-1)  # convert specs to (100, 256, 256, 1)
        # print("-------------------", specs.shape)
        if group_label != None:
            labels = h5file[group_label][batchcount:batchcount + batchsize].data

        batchcount += 1
        if batchcount % 10 == 0:
            print("batchcount=", batchcount, "batchsize=", batchsize)
        if AE:
            yield (specs, specs)  # Note Bene - for auto encoder target is input!
        else:
            yield (specs, labels)


####################################################################################
def makeEncoderplot(filename, spec, encoder, specPred, lbl, wavfilename):
    print(filename, spec.shape, encoder.shape)
    n_images = 9  # 8 encoder and 0ne input spectrogram
    fig, axes = plt.subplots(3, 4, figsize=(12, 13))  # plt.figure(8,3,figsize=(4, 3))

    fig.suptitle("{}_array from\n {}\nlabel {}".format(filename, wavfilename, lbl))
    for i in range(2):  # rows
        for j in range(4):  # cols
            k = i * 4 + j
            if k < 8:
                encary = encoder[:, :, k]
                axes[i, j].imshow(encary)
                axes[i, j].xaxis.set_visible(False)
                axes[i, j].yaxis.set_visible(False)
    axes[2, 1].imshow(spec)
    axes[2, 2].imshow(specPred)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()


#####################################################################################
np.random.seed(42)  # sml db leads off with some orca calls

h5filename = "/home/val/PyCharmFiles/TF_val/AEinput_8_28/OS_09_12_21SpecsNormedSquared_1_wavs.h5"
h5filename = "/home/val/PyCharmFiles/TF_val/AEinput_8_24/OS_09_12_21SpecsNormedSquared_11_wavs.h5"
h5filename = "../h5files/OS_09_12_21SpecsNormedSquared_1_wavs.h5"

plotDir = d3.checkDir("../outputFiles/AE_plots/")
AE_weights = "../outputFiles/AE_weights/conv_ae_09_12_21_passby_6_NN_6_epochs_2000_2000fullNNwts"

h5db = h5py.File(h5filename, 'r')

print('Database keys', h5db.keys())
for k in h5db.attrs.keys():
    print(f"     attribute: {k} => {h5db.attrs[k]}")
for key in h5db.keys():
    print(key, h5db[key])
    for k in h5db[key].attrs.keys():
        print(f"          attribute: {k} => {h5db[key].attrs[k]}")
    for key2 in h5db[key].keys():
        print("    ",key2,  h5db[key][key2])

parmstrDict = eval(h5db.attrs['paramStr'])  # converts string back to dict
thisWav = 'wav_1'
encoders = h5db['{}/encoders'.format(thisWav)]
specs = h5db['{}/specs'.format(thisWav)]
wavfilename = h5db['{}'.format(thisWav)].attrs['filename']

print(encoders.shape)
print(specs.shape)
initialIdx = 0
nPlots = 30
step = 1

ndim = 256
conv_ae = AE_6()
conv_ae.load_weights(AE_weights)
enclist = []
for i in range(nPlots):
    idx = i * step + initialIdx
    enclist.append(encoders[idx][4])

decodedlist = conv_ae.decoder.predict(np.expand_dims(enclist, axis=-1))

#
# print("\n", h5file, "keys are", h5file.keys())
# for key in h5file.keys():
#     print(key, "length", len(h5file[key]))
#
# #evalAry = generate_AE_arrays_from_h5(enc_h5file, True, 'eval_specs', None, batchsize)
# #evalspecs = next(evalAry)
#
# arryin = spec_h5file['train_specs']
# encoded = enc_h5file['train_specs']
#
# print(len(arryin))
# print(len(encoded))
#

plotname = wavfilename.split(".")[0]
for i in range(nPlots):
    idx = i * step + initialIdx
    lbl = specs[idx][1]
    spec = specs[idx][4]    # 4 has the arrays
    enc = enclist[i]
    specPred = decodedlist[i]
    print("plot", wavfilename, " at idx", idx)
    if idx == 0:
        print(spec)
    makeEncoderplot(plotDir + "{}_at_{}_s__8_27_encoded_db.jpg".\
                    format(plotname, idx*parmstrDict['DeltaT']), spec, enc, specPred, lbl, wavfilename)
