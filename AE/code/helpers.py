#!/usr/bin/python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import math
from random import random
import numpy as np
import copy
# from tensorflow import keras
# import tensorflow as tf
# from tensorflow.keras import Model, Input
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose
# from tensorflow.keras.layers import ReLU, BatchNormalization
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras import layers

import pickle
###################################################  for WAV processing
metadata = {}
metadata['Tspec'] = 3
metadata['DeltaT'] = 1
metadata['Ntimes'] = 256
metadata['Nfreqs'] = 255    # this MUST be one smaller than desired number of freq values
metadata['f_low'] = 300     #    as the total power is pushed into the bottom making 128
metadata['f_high'] = 10000
metadata['Nfft'] = 512
metadata['logscale'] = False
DEBUG = 0

def checkDir(theDir):  # make sure directory has a / at the end
    items = theDir.split('/')
    if items[-1] != '':
        theDir += '/'
    return theDir

def makeDir(thisdir):
    try:
        os.mkdir(thisdir)
        return thisdir
    except:
        print('-------------Already have ', thisdir)
        return thisdir

def getWavs(wavDir):
    wavfilelist = []
    try:
        os.chdir(wavDir)
        # get list of the wav or WAV files
        wavfilelist = [f for f in os.listdir('.') if
                       any(f.endswith(ext) for ext in ['WAV', 'wav'])]  # os.listdir()  # get list of the wav files
        wavfilelist.sort()
    except:
        print("failed to load wavs from dir ", wavDir)
        exit()
    return wavfilelist
def compressPsdSliceLog(freqs, psds, flow, fhigh, nbands, doLogs):
    compressedSlice = np.zeros(nbands + 1)  # totPwr in [0] and frequency of bands is flow -> fhigh in nBands steps
    #    print("Num freqs", len(freqs))
    idxPsd = 0
    idxCompressed = 0
    fbands = setupFreqBands(flow, fhigh, nbands, doLogs)
    dfbands = []
    for i in range(len(fbands)-1):
        df = fbands[i+1] - fbands[i]
        dfbands.append(df)
    dfbands.append(df)   # add one more to have 1 for 1 with fbands
    # integrate psds into fbands
    df = freqs[1] - freqs[0]   # this freq scale is linear as it comes from the wav samplerate
    totPwr = 0
    while freqs[idxPsd] <= fhigh and idxCompressed < nbands:
        # find index in freqs for the first fband
        inNewBand = False
        if DEBUG == 10: print(freqs[idxPsd] , fbands[idxCompressed])
        while freqs[idxPsd] < fbands[idxCompressed]:  # step through psd frequencies until greater than this fband
            idxPsd += 1
            inNewBand = True
        deltaf = freqs[idxPsd] - fbands[idxCompressed]  # distance of this psd frequency into this fband
        if DEBUG == 10: print(deltaf)
        if deltaf > dfbands[idxCompressed]:  # have jumped an entire band
            compressedSlice[idxCompressed + 1] += psds[idxPsd]*dfbands[idxCompressed]/df  # frac of psds = slice
            idxCompressed += 1
        else:
            pfrac = deltaf / df
            compressedSlice[idxCompressed + 1] += pfrac * psds[idxPsd]  # put frac of first pwr in psd
            if inNewBand:
                idxCompressed += 1
            idxPsd += 1
        if DEBUG == 10: print(idxPsd, idxCompressed, deltaf, inNewBand)
        if DEBUG == 10: print("")
    compressedSlice[0] = np.sum(compressedSlice)
    return compressedSlice

def compressPsdSliceLogOLD(freqs, psds, flow, fhigh, nbands, doLogs):
    compressedSlice = np.zeros(nbands + 1)  # totPwr in [0] and frequency of bands is flow -> fhigh in nBands steps
    #    print("Num freqs", len(freqs))
    idxPsd = 0
    idxCompressed = 0
    fbands = setupFreqBands(flow, fhigh, nbands, doLogs)
    # integrate psds into fbands
    df = (fhigh - flow) / nbands
    totPwr = 0
    while freqs[idxPsd] <= fhigh and idxCompressed < nbands:
        # find index in freqs for the first fband
        while freqs[idxPsd] < fbands[idxCompressed]:  # step through psd frequencies until greater than this fband
            idxPsd += 1
        dfband = freqs[idxPsd] - fbands[idxCompressed]  # distance of this psd frequency into this fband
        pfrac = 1 - dfband / df
        psd = pfrac * psds[idxPsd]  # put frac of first pwr in psd
        fmax = fhigh
        if idxCompressed < nbands - 1:
            fmax = fbands[idxCompressed + 1]
        while freqs[idxPsd] < fmax:
            psd += psds[idxPsd]
            idxPsd += 1
        dfband = freqs[idxPsd] - fmax
        pfrac = dfband / df
        psd += pfrac * psds[idxPsd]
        compressedSlice[idxCompressed + 1] = psd
        if DEBUG > 0:
            print(idxCompressed+1, psd)
        totPwr += psd
        idxCompressed += 1
    compressedSlice[0] = totPwr
    return compressedSlice

def setupFreqBands(flow, fhigh, nbands, doLogs):
    df = (fhigh - flow) / nbands
    fbands = np.zeros(nbands)
    if not doLogs:
        for i in range(nbands):
            fbands[i] = flow + i*df
    else:
        dlogf = (np.log10(fhigh) - np.log10(flow)) / (nbands - 0)
        fbands[0] = flow
        for i in range(1, nbands):
            if DEBUG > 0:
                print("np.power(10,(i * dlogf))", np.power(10,(i * dlogf)))
            fbands[i] = np.power(10,np.log10(flow) + (i * dlogf))
        if DEBUG > 0:
            print("flow,fbands,fhigh",flow,fbands,fhigh)
    return fbands



###################################################
def saveObject(obj, path, filename):
    save_obj(obj, path+filename)

def printConfusionMatrix(kerasModel, datagroup, x_data, y_data):
    print(datagroup, "dataset........", x_data.shape)

    predictions = kerasModel.predict(x_data)  # feed the numpy arrays to predict
    confMatrix = np.zeros([2, 2])  # rows are actual values
    for i in range(len(predictions)):  # cols are predictions
        pred = predictions[i][0]    #  Note Bene  WHY DOES PREDICTIONS HAVE THIS STRUCTURE??
        lbl = y_data[i]
#        print(lbl, pred)
        if lbl == 0:  # a row of true backgrounds
            if pred < 0.5:
                confMatrix[0, 0] += 1  # True negative
            else:
                confMatrix[0, 1] += 1  # False negative
        if lbl == 1:  # a row of true signals
            if pred < 0.5:
                confMatrix[1, 0] += 1  # False positive
            else:
                confMatrix[1, 1] += 1  # True positive
    recall    =  confMatrix[1,1]/(confMatrix[1,1] + confMatrix[0,1])
    precision =  confMatrix[1,1]/(confMatrix[1,1] + confMatrix[1,0])

    print("Confusion matrix fractions for predictions on dataset\n             ", datagroup, "of length", len(x_data))
    confMatrix = confMatrix / np.sum(confMatrix)  # normalize to fractions
    print('             PREDICT   0            1')
    print('   Label = 0      TN {:0.3f}    FN {:0.3f}'.format(confMatrix[0,0], confMatrix[0,1]))
    print('   Label = 1      FP {:0.3f}    TP {:0.3f}'.format(confMatrix[1,0], confMatrix[1,1]))
    print("Precision is {:0.3f}   Recall is {:0.3f}".format(precision, recall))
    print("Precision = frac of + predictions that are correct.")
    print("Recall    = frac of actual + that are predicted.\n")


def drawline(istart, jstart,ary, theta):
    ndim = ary.shape[0]
    linelen = int(random()*ndim/2 + ndim//2)
#    print("istart, jstart, linelen, theta", istart, jstart, linelen, theta)
    for i in range(linelen):
        ix = istart + int(math.cos(theta)*i)
        iy = jstart + int(math.sin(theta)*i)

        if ix<3: ix = 3
        if ix >=ndim-3: ix = ndim-3
        if iy<3: iy = 3
        if iy >=ndim-3: iy = ndim-3
        ary[ix-2, iy] = math.sqrt(random())
        ary[ix+2, iy] = math.sqrt(random())
        ary[ix, iy-2] = math.sqrt(random())
        ary[ix, iy+2] = math.sqrt(random())        
                
        ary[ix-1, iy] = math.sqrt(random())
        ary[ix+1, iy] = math.sqrt(random())
        ary[ix, iy-1] = math.sqrt(random())
        ary[ix, iy+1] = math.sqrt(random())
        ary[ix, iy] = 1
    return ary

def getRandArrays(Narys, ndim):
    x_new = []
    theta = 0
    for i in range(Narys):
        ary = np.random.rand(ndim,ndim)   # initialize array
        for _ in range(2):
            istart = ndim//2 + int((random()-0.5)*ndim//3)
            for _ in range(2):
                jstart = ndim//2 + int((random()-0.5)*ndim//3)
                ary = drawline(istart, jstart, ary,theta)
                theta += 1.1
#                print("i=", i, istart, jstart, theta, ndim)
#        plt.imshow(ary)
#        plt.savefig("outputs/"+"arry_{}".format(i)) 
#        input("???")
        x_new.append(ary)
    return np.asarray(x_new)   
    
    
def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

def save_fig(plt, img_path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(img_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    
def show_reconstructions(model, images, img_path, filename, n_images=5, tight_layout=True, fig_extension="png", resolution=300):
#    print("helpers show_reconstruction inputs ", images[0],"\n", n_images, filename)
    reconstructions = model.predict(images[0][:n_images])
#    print("-got reconstructions")
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    plt.title(filename)
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[0][image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
    path = os.path.join(img_path, filename + "." + fig_extension)
    print("Saving figure", filename)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)  

def show_reconstructionsSept(model, images, img_path, filename, n_images=5, tight_layout=True, fig_extension="png", resolution=300):
    print("::::::::::::::", images[0])
    img = []
    for i in range(n_images):
        print(type(images))

#        img.append(itm[0])

    print("helpers show_reconstruction inputs ",img[0], "\n", n_images, filename)
    input("[[[[[[[[[[[")
    reconstructions = model.predict(img)  # grab first element, the ary
    print("\n--------------------------------got reconstructions")
    input(";;;;;;;;;;;;;;;;;;;;;")
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    plt.title(filename)
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(img[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
    path = os.path.join(img_path, filename + "." + fig_extension)
    print("Saving figure", filename)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)  
    plt.close()
    
def show_history(model, history, N, score, totEpochs, img_path, filename):
    print("history keys=", history.history.keys())
    keylist = []
    for key in history.history.keys():
        keylist.append(key)
    print("List of keys", keylist)
    plt.close("all")
    for key in keylist:
        plt.plot(history.history[key])

    title = '{} history and len(X_train)={} with loss/accuracy {:.3f}/{:.3f}\nTotal epochs {}'.format(model, N, score[0], score[1], totEpochs)
    plt.title(title)
    plt.ylabel('amount')
    plt.xlabel('epoch')
    plt.legend([keylist[0], keylist[1]], loc='upper left')
    plt.savefig(img_path + filename)  
    plt.close() 

#####################################################
def plot_percent_hist(ax, data, bins):
    counts, _ = np.histogram(data, bins=bins)
    widths = bins[1:] - bins[:-1]
    x = bins[:-1] + widths / 2
    ax.bar(x, counts / len(data), width=widths*0.8)
    ax.xaxis.set_ticks(bins)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda y, position: "{}%".format(int(np.round(100 * y)))))
    ax.grid(True)
def plot_activations_histogram(model, X_valid, img_path, filename, height=1, n_bins=10, tight_layout=True, fig_extension="png", resolution=300):
    X_valid_codings = model.predict(X_valid) #.numpy()
    activation_means = X_valid_codings.mean(axis=0)
    mean = activation_means.mean()
    bins = np.linspace(0, 1, n_bins + 1)

    fig, [ax1, ax2] = plt.subplots(figsize=(10, 3), nrows=1, ncols=2, sharey=True)
    plot_percent_hist(ax1, X_valid_codings.ravel(), bins)
    ax1.plot([mean, mean], [0, height], "k--", label="Overall Mean = {:.2f}".format(mean))
    ax1.legend(loc="upper center", fontsize=14)
    ax1.set_xlabel("Activation")
    ax1.set_ylabel("% Activations")
    ax1.axis([0, 1, 0, height])
    plot_percent_hist(ax2, activation_means, bins)
    ax2.plot([mean, mean], [0, height], "k--")
    ax2.set_xlabel("Neuron Mean Activation")
    ax2.set_ylabel("% Neurons")
    ax2.axis([0, 1, 0, height])
    path = os.path.join(img_path, filename + "." + fig_extension)
    print("Saving figure", filename)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)    
        

#####################################################
# K = keras.backend
# kl_divergence = keras.losses.kullback_leibler_divergence
#
# class KLDivergenceRegularizer(keras.regularizers.Regularizer):
#     def __init__(self, weight, target=0.1):
#         self.weight = weight
#         self.target = target
#     def __call__(self, inputs):
#         mean_activities = K.mean(inputs, axis=0)
#         return self.weight * (
#             kl_divergence(self.target, mean_activities) +
#             kl_divergence(1. - self.target, 1. - mean_activities))



