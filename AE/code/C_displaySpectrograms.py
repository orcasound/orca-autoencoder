import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import h5py
import helpers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.models import load_model



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

def checkSpecs(spec, enc):
    match = True
    for i in range(4):
        if spec[i] != enc[i]:
            match = True
            print("item ", i, "error", spec[i], "!=", enc[i])
    return match


#######################################################################################

#######################################################################################

h5in_1filename = "../h5files/wavsAudacityLabeled_4wavs.h5"
h5in_1filename = "../h5files/wavsAudacityLabeled_4wavsNNencodings_1.h5"
h5in_1 = h5py.File(h5in_1filename, 'r')

spectrograms = h5in_1['audio1']['spectrograms']
encoders = h5in_1['audio1']['encodings']
specs = []
encs  = []
lbls  = []
wavs  = []
idxs  = []
preds = []
Nlabeled = 0
i=0
while len(specs) < 100:
    if checkSpecs(spectrograms[i], encoders[i]):
        expertlbl = spectrograms[i][2]
        parm = spectrograms[i][4]
        if parm < 0 and parm > -1.0:
            wavs.append(spectrograms[i][0])
            idxs.append(spectrograms[i][1])
            lbls.append((spectrograms[i][2], spectrograms[i][3], spectrograms[i][4]))
            specs.append(spectrograms[i][5])
            encs.append(encoders[i][5])

    # else:
    #     print("Mismatch in record ", i)
    i = i+1

# Run AE on specs for comparisions
modelFilenameWts =  "../outputFiles/AE_weights/conv_ae_test_10_min_9_18h5_NN_6_epochs_2110_3110"
modelClass = "AE_6()"
conv_ae = helpers.AE_6()
conv_ae.load_weights(modelFilenameWts)
preds = conv_ae.decoder.predict(np.expand_dims(encs, axis=-1))

print(conv_ae.encoder.summary())
print(conv_ae.decoder.summary())

for i in range(len(specs)):
    filename = "../outputFiles/specPlotsEncRelabel_1/enc_plot_lbl_{}_{}.jpg".format(lbls[i][0], i)
    makeEncoderplot(filename, specs[i], encs[i], preds[i], lbls[i], wavs[i])


