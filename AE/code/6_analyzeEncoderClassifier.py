import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math
import h5py
import random
import helpers as h
import time
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
import tensorflow as tf
import helpers as d3
"""
  Access h5 database of spectrograms
  examine classifier predictions

"""
#######################################################################
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
########################################################################
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

########################################################################
np.random.seed(42)

h5Inputfilename = "../h5files/OS_09_12_21SpecsNormedSquared_1_wavs.h5"
h5ID = "1_wav_h5"
h5Inputfilename = "../h5files/OS_09_12_21SpecsNormedSquared_11_wavs.h5"
h5ID = "11_wav_h5"

AE_weights = "../outputFiles/AE_weights/conv_ae_09_12_21_passby_6_NN_6_epochs_2000_2000fullNNwts"
plotDir = d3.makeDir("../outputFiles/AE_plots/{}/".format(h5ID))


# plot the first records
nPlots = 10  # max num of each type of plot
step = 1
initialIdx = 0

ndim = 256
conv_ae = AE_6()
conv_ae.load_weights(AE_weights)


h5db = h5py.File(h5Inputfilename, 'a')
print(h5Inputfilename, ' Database keys ', h5db.keys())
for k in h5db.attrs.keys():
    print(f"     attribute: {k} => {h5db.attrs[k]}")
for key in h5db.keys():
    print(key, h5db[key])
    for k in h5db[key].attrs.keys():
        print(f"          attribute: {k} => {h5db[key].attrs[k]}")
    for key2 in h5db[key].keys():
        print("    ",key2,  h5db[key][key2])
parmstrDict = eval(h5db.attrs['paramStr'])

FPlist = []
FNlist = []
encListFull = []
specListFull = []
wavFilenames = []
recCnt = 0
classifiedCnt = 0
for wav in h5db.keys():
    thisWavEncs = h5db[wav]['encoders']
    thisWavSpecs = h5db[wav]['specs']
    wf = h5db[wav].attrs['filename']
    i = 0
    for enc in thisWavEncs:
        lbl = enc[1]
        lblp = enc[2]
        if lblp != -1:
            classifiedCnt += 1
        parm = enc[3]
#        print(lbl, lblp, "{:0.3f}".format(parm))
        if lbl == 0 and lblp == 1:
            FPlist.append((wav, enc))
        if lbl == 1 and lblp == 0:
            FNlist.append((wav, enc))
        encListFull.append((wav, enc))
        specListFull.append((wav,thisWavSpecs[i][4]))
        wavFilenames.append(wf)
        i += 1
        recCnt += 1

print("of", recCnt, "records:", classifiedCnt, "were classifid with len(FPlist)",len(FPlist), "len(FNlist)", len(FNlist))

enclist = []
speclist = []
labels = []
for i in range(nPlots):
    idx = i * step + initialIdx
    enclist.append(encListFull[idx][1][4])
    speclist.append(specListFull[idx][1])
    lbl = "idx {}, {}, ({} {:0.3f})".format(encListFull[idx][1][0], encListFull[idx][1][1], encListFull[idx][1][2], encListFull[idx][1][3], encListFull[idx][1][4])
    labels.append(lbl)
declist = conv_ae.decoder.predict(np.expand_dims(enclist, axis=-1))

for i in range(len(enclist)):
    wavfilename = wavFilenames[i]
    makeEncoderplot(plotDir + "{}_encPlot__{}.jpg".format(h5ID, i), speclist[i], enclist[i], declist[i], labels[i], wavfilename)



# plot the false positives
encs = np.asarray(FPlist)[:,1]
enclist = []
for enc in encs:
    enclist.append(enc[4])
enclist = np.asarray(enclist)
decodedlist = conv_ae.decoder.predict(np.expand_dims(enclist, axis=-1))

for i in range(len(FPlist)):
    wav = FPlist[i][0]
    idx = FPlist[i][1][0]
    spec = h5db[wav]['specs'][idx][4]
    enc = FPlist[i][1][4]
    lbl = FPlist[i][1][1]
    lblp = FPlist[i][1][2]
    parm = FPlist[i][1][3]
    newlbl = "idx {},  {}, ({} {:0.3f})".format(idx, lbl, lblp, parm)
    saveFilename = "{}_FP_{}.jpg".format(h5ID, i)
    dec = decodedlist[i]
    wavfilename = h5db[wav].attrs['filename']
    makeEncoderplot(plotDir + saveFilename, spec, enc, dec, newlbl, wavfilename)
    i += 1
    if i > nPlots:
        break

encs = np.asarray(FNlist)[:,1]
enclist = []
for enc in encs:
    enclist.append(enc[4])
enclist = np.asarray(enclist)
decodedlist = conv_ae.decoder.predict(np.expand_dims(enclist, axis=-1))
i = 0
for rec in FNlist:
    wav = rec[0]
    idx = rec[1][0]
    spec = h5db[wav]['specs'][idx][4]
    enc = rec[1][4]
    lbl = rec[1][1]
    lblp = rec[1][2]
    parm = rec[1][3]
    newlbl = "idx {},  {}, ({} {:0.3f})".format(idx, lbl, lblp, parm)
    saveFilename = "{}_FN_{}.jpg".format(h5ID, i)
    dec = decodedlist[i]
    wavfilename = h5db[wav].attrs['filename']
    makeEncoderplot(plotDir + saveFilename, spec, enc, dec, newlbl, wavfilename)
    i += 1
    if i> nPlots:
        break