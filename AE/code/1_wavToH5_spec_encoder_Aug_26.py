#!/usr/bin/python3



import sys
import helpers as d3
import importlib
importlib.reload(d3)
import copy

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import math
import time
import random 
import datetime
import soundfile as sf

from tensorflow.keras import Model
from tensorflow.keras import layers
import tensorflow as tf

####################################### python routines
def decodeWAV(idx, filelist):
    filename = filelist[idx]
    items = filename.split('_')
    site = items[0]
    mo = int(items[1])
    dy = int(items[2])
    yr = int(items[3])
    hr = int(items[4])
    mn = int(items[5])
    sc = int(items[6])
    x = datetime.datetime(yr, mo, dy, hr, mn, sc)
#    y = x.timestamp()
    return x, filename
def calculateSpectrogram(WAVs, startsecs, wavStartIdx,  specduration):
    Ntimes = d3.metadata['Ntimes']
    f_low = d3.metadata['f_low']
    f_high = d3.metadata['f_high']
    Nfreqs = d3.metadata['Nfreqs']
    logscale = d3.metadata['logscale']
    Nwavs = len(WAVs)
    with sf.SoundFile(WAVs[0]) as f:
        blocksize = int(specduration * f.samplerate // Ntimes)  # this is one bin in the final spectrogram
        samplerate = f.samplerate
    samples, wavStartIdx = getSamples(startsecs, wavStartIdx, int(specduration * f.samplerate), WAVs)
#    print("wavStartIdx=",wavStartIdx)
    if wavStartIdx < 0:
        return 0,0,wavStartIdx
    spectrogram, specCompressed = getCompressedSpectrogram(Ntimes, Nfreqs, f_low, f_high, logscale, samplerate, samples)
    return spectrogram, specCompressed, wavStartIdx, samplerate

def getSamples(startsecs, wavStartIdx, Nsamples, WAVs):
    # need to get Ntimes blocks of time series data
    channelchoice = -1    # pick channel with higher amplitude
    typedict = {}
    typedict['FLOAT'] = 'float32'
    typedict['PCM_16'] = 'int16'

    NsamplesNeeded = Nsamples
    npsamples = []
    while NsamplesNeeded > 0:
        if wavStartIdx >= len(WAVs):
            return npsamples, -1
        with sf.SoundFile(WAVs[wavStartIdx]) as f:
#            print("-------------reading wav file", WAVs[wavStartIdx], "wavStartIdx", wavStartIdx)
            availableSamples = f.seek(0, sf.SEEK_END) - int(startsecs*f.samplerate)
#            if availableSamples < 0:
#                print(startsecs)
#            print("availableSamples=", availableSamples, WAVs[wavStartIdx])
            if len(npsamples) == 0:  # test for first wav file only
                if availableSamples > 0:
                    f.seek(int(startsecs*f.samplerate))   # for first wav file, start at desired number of secs into file
                else:
                    f.seek(0)  # start at beginning of wav file, continuing into a new file
            while availableSamples > 0 and NsamplesNeeded > 0:
                if availableSamples >= NsamplesNeeded:
                    data = f.buffer_read(NsamplesNeeded, dtype=typedict[f.subtype])
                    npdata = convertToNumpy(f, typedict, data)
                    NsamplesNeeded = 0
                else:
                    data = f.buffer_read(availableSamples, dtype=typedict[f.subtype])
                    npdata = convertToNumpy(f, typedict, data)
                    NsamplesNeeded  -= availableSamples
                    startsecs = 0
                    availableSamples = 0
                    wavStartIdx += 1     # setup for next wav file
                    if wavStartIdx >= len(WAVs):
                        break   # we are at the end of all the WAV files
                if len(npsamples) == 0:
                    npsamples = npdata
                else:
                    npsamples = np.append(npsamples, npdata)


            f.close()

#    print("n samples", len(npsamples))
    return npsamples, wavStartIdx

def getCompressedSpectrogram(Ntimes, Nfreqs, f_low, f_high, logscale, samplerate,
                      samples):  # specBlock will be 1-D spectrograms, one for each slice in time
    specGram = []
    samplesPerBin = len(samples) // Ntimes
    for i in range(Ntimes):
        data = samples[i*samplesPerBin: (i+1)*samplesPerBin]
        data = data * np.hamming(len(data))
#        data = data * np.blackman(len(data))
#         plt.plot(data)
#         plt.show()
#         plt.close()
        spec = np.abs(np.fft.rfft(data))  ############, 4096)) #Nfft))
        f_values = np.fft.fftfreq(len(data), d=1. / samplerate)
        # if i == 70 or i == 1:
        #     plt.plot(np.log10(spec))
        #     plt.show()
        #     plt.close()
        spec = d3.compressPsdSliceLog(f_values, spec, f_low, f_high, Nfreqs, logscale)
        # if i == 70 or i == 1:
        #     plt.plot(np.log10(spec))
        #     plt.show()
        #     plt.close()
        specGram.append(spec)  # flip to put low frequencies at 'bottom' of array as displayed on screen
        # if i%32 == 0:
        #     print("i=",i)

    #  transform array
    # plt.plot(np.log10(specGram[20]))
    # plt.show()
    # plt.close()
    # plt.imshow(np.log10(specGram))
    # plt.show()
    # plt.close()
    specGram = np.log10(np.flip(specGram) + 0.001)  # to avoid log(0)
    specGram = np.rot90(specGram, 3)  ###COULD use  square roots etc to bring lower peaks up  i.e.  0.36  -> 0.6  -> 0.77
    # plt.imshow(specGram)
    # plt.title("Un-normalized")
    # plt.gray()
    # plt.show()
    # pmax = np.max(specGram)
    # pmin = np.min(specGram)
    # specGram = (specGram - pmin) / (pmax - pmin + 0.001)  ###  Normalize to 0 -> 1
    # # plt.imshow(specGram)
    # # plt.title("Min-Max Normalized")
    # # plt.gray()
    # # plt.show()
    specGramNorm = np.asarray(getNorm(copy.copy(specGram)))
    # plt.imshow(specGramNorm)
    # plt.title("Normalized to mean +- 2 sd")
    # plt.gray()
    # plt.show()
    # plt.imshow(np.square(specGramNorm))
    # plt.title("Normalized to mean +- 2 sd then squared")
    # plt.gray()
    # plt.show()
    # input('???????')
    return specGram, np.square(specGramNorm)

def getNorm(ary):
    nrows = ary.shape[1]
    bbmax = np.max(ary[nrows-1, :])    # get max of bottom row
    bbmin = np.min(ary[nrows - 1, :])
    ary[nrows-1, :] = (ary[nrows-1, :] - bbmin)/(bbmax - bbmin)         # normalize the bottom row to 0 -> 1
    aryMean = np.mean(ary[0:nrows - 1, :])
    aryStd  = np.std(ary[0:nrows - 1, :])
    ary[0:nrows - 1, :] = (ary[0:nrows - 1, :] - aryMean)/(4 * aryStd)
    ary[ary<-1] = -1
    ary[ary>1] = 1
    ary = ary/2.0 + 0.5
    return ary
def convertToNumpy(f, typedict, data):
    channelchoice = -1   #  -1 to pick channel with higher amplitude
    if f.channels == 2:
        if channelchoice == -1:
            ch0 = np.average(np.abs(np.frombuffer(data, dtype=typedict[f.subtype])[0::2]))
            ch1 = np.average(np.abs(np.frombuffer(data, dtype=typedict[f.subtype])[1::2]))
            if ch0 > ch1:
                channelchoice = 0
            else:
                channelchoice = 1
        npdata = np.frombuffer(data, dtype=typedict[f.subtype])[channelchoice::2]
    else:
        npdata = np.frombuffer(data, dtype=typedict[f.subtype])
    return npdata

def buildDatasets(h5db, nTimes, nFreqs, speclist, augment):
    trainlist = []
    testlist = []
    evallist = []
    for spec in speclist:
        rnd = random.random()
        if rnd < 0.75:
            trainlist.append(spec)
        else:
            if rnd < 0.9:
                testlist.append(spec)
            else:
                evallist.append(spec)
    if not augment:
        h5db.create_dataset('train_specs', data=trainlist, compression=True, chunks=True, maxshape=(None, nTimes, nFreqs))
        h5db.create_dataset('test_specs', data=testlist, compression=True, chunks=True, maxshape=(None, nTimes, nFreqs))
        h5db.create_dataset('eval_specs', data=evallist, compression=True, chunks=True, maxshape=(None, nTimes, nFreqs))
    else:
        h5db['train_specs'].resize((h5db['train_specs'].shape[0] + len(trainlist)), axis=0)
        h5db['train_specs'][-len(trainlist):] = trainlist
        h5db['test_specs'].resize((h5db['test_specs'].shape[0] + len(testlist)), axis=0)
        h5db['test_specs'][-len(testlist):] = testlist
        h5db['eval_specs'].resize((h5db['eval_specs'].shape[0] + len(evallist)), axis=0)
        h5db['eval_specs'][-len(evallist):] = evallist

def buildEncodedDatasets(h5db, wavCnt, wavfilename, samplerate, h5filename, paramStr, speclist, encodedlist, augment):
     print("in build encoded database", wavfilename)
     group = 'wav_{}'.format(wavCnt)  # this will be unique index, expert label, model label, model parm threshold
     spec_datatype = np.dtype([('idx', '<i8'), ('lbl', '<i8'), ('lblp', '<i8'), ('parm', '<f8'), ('ary', '<f8', (256, 256))])
     enc_datatype  = np.dtype([('idx', '<i8'), ('lbl', '<i8'), ('lblp', '<i8'), ('parm', '<f8'), ('ary', '<f8', (16, 16, 8))])
     if not augment:  # initializing
         specobj = []
         encobj = []
         for i in range(len(speclist)):
             specobj.append((i, -1, -1, -1.0, speclist[i]))
             encobj.append((i, -1, -1, -1.0, encodedlist[i]))
         h5db.create_group(group)
         h5db[group].create_dataset('specs', data=specobj, maxshape=(None, ), dtype=spec_datatype)
         h5db[group].create_dataset('encoders', data=encobj, maxshape=(None,), dtype=enc_datatype)
         # h5db[group].create_dataset('specs',  compression=True, chunks=True, maxshape=(None, val_datatype),\
         #                dtype=val_datatype)

         # h5db[group].create_dataset('encoders', data=encobj, compression=True, chunks=True, maxshape=(None, val_datatype),\
         #                dtype=val_datatype)
         h5db[group].attrs['filename'] = wavfilename
         h5db[group].attrs['samplerate'] = samplerate
     else:
         Nindb = len(h5db['{}/specs'.format(group)])
         specobj = []
         encobj = []
         for i in range(0, len(speclist)):
             specobj.append((i + Nindb, -1, -1, -1.0, speclist[i]))
             encobj.append((i + Nindb, -1, -1, -1.0, encodedlist[i]))

         h5db['{}/specs'.format(group)].resize((h5db['{}/specs'.format(group)].shape[0] + len(specobj)), axis=0)
         h5db['{}/specs'.format(group)][-len(speclist):] = specobj
         h5db['{}/encoders'.format(group)].resize((h5db['{}/encoders'.format(group)].shape[0] + len(encobj)), axis=0)
         h5db['{}/encoders'.format(group)][-len(encobj):] = encobj
     h5db.flush()

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

#######################################

wavDir = d3.checkDir('../wavFiles/')
h5SaveDir = d3.checkDir('../h5files/')

h5filename = "OS_09_12_21SpecsNormedSquared_1_wavs.h5"   # MAKE SURE THIS IS THE NAME YOU WANT FOR SAVING THE NEW DATABASE
try:
    h5db = h5py.File(h5SaveDir + h5filename, mode='a')
except:
    print("File {} already exists.  Delete it".format(h5SaveDir + h5filename))
    quit()
d3.makeDir(h5SaveDir) # create this directory if it doesn't exist
trainSpecs = []
trainLbls = []
testSpecs = []
testLbls = []
evalSpecs = []
evalLbls = []

Tspec = d3.metadata['Tspec']
DeltaT = d3.metadata['DeltaT']
f_low = d3.metadata['f_low']
f_high = d3. metadata['f_high']
Ntimes = d3.metadata['Ntimes']
Nfreqs = d3.metadata['Nfreqs']
Nfft = d3.metadata['Nfft']
logscale = d3.metadata['logscale']

wavfilelist = d3.getWavs(wavDir)
print("\n------------------------------number of wavs is", len(wavfilelist))
print("first wav is ", wavfilelist[0])

print('Database keys', h5db.keys())
for key in h5db.keys():
    print(h5db[key])
    for k in h5db[key].attrs.keys():
            print(f"attribute: {k} => {h5db[key].attrs[k]}")
    for key2 in h5db[key].keys():
        print("    ", h5db[key][key2])
        for k in h5db[key][key2].attrs.keys():
            print(f"          attribute: {k} => {h5db[key].attrs[k]}")


dsetBlock = 10
#  encoder

modelFilenameWts = "conv_ae_09_12_21_passby_6_NN_6_epochs_2000_2000" + "fullNNwts"
modelClass = "AE_6()"
ndim = 256
conv_ae = AE_6()
conv_ae.load_weights("/home/val/PyCharmFiles/TF_val/data/" + modelFilenameWts)

paramDict = d3.metadata
paramStr = str(paramDict)
h5db.attrs['paramStr'] = paramStr
h5db.attrs['modelFilenameWts'] = modelFilenameWts
h5db.attrs['modelClass'] = modelClass

# process all the files in the wavDir
alldone = False
startSecs = wavFileStartPtr = priorPtr = 0
cnt = 0
runStart = time.time()
speclist = []
wavfileCnt = 0
totRecords = 0
priorWavfile = ""
tstart = time.time()

while not alldone:
    startDatetime, wavfilename = decodeWAV(wavFileStartPtr, wavfilelist)
    cnt += 1
#    print(wavfilelist, startSecs, wavFileStartPtr, Tspec)
    try:
        spectrogram, spectrogramNormed, wavFileStartPtr, samplerate = calculateSpectrogram(wavfilelist, startSecs, wavFileStartPtr, Tspec)
        speclist.append(spectrogramNormed)
        totRecords += 1
#        print(cnt, totRecords)
        if cnt % dsetBlock == 0 or wavfilename != priorWavfile:
            #  encode this dsetBlock of spectrograms
            encodedlist = conv_ae.encoder.predict(np.expand_dims(speclist, axis=-1))
            # save spectrograms to h5db
            print("records processed =", totRecords)
            if wavfilename != priorWavfile:  # first time for this wav file
                priorWavfile = wavfilename
                wavfileCnt += 1

                buildEncodedDatasets(h5db, wavfileCnt, wavfilename, samplerate, h5filename, paramStr, speclist, encodedlist, False)  # False signifies need to create databases
            else:
                buildEncodedDatasets(h5db, wavfileCnt, wavfilename, samplerate, h5filename, paramStr, speclist, encodedlist, True)
            speclist = []
        # print(" looping------",wavfileCnt, cnt, wavfilename, wavFileStartPtr)
        # print(h5db['train_specs'].shape)
        # print(h5db['test_specs'].shape)
        # print(h5db['eval_specs'].shape)


    except:
        print("got an error with", wavfilelist, startSecs, wavFileStartPtr, Tspec)
        if wavFileStartPtr == len(wavfilelist) - 1:
            break
    if wavFileStartPtr < 0:
        alldone = True
    else:
        if wavFileStartPtr == priorPtr:
            startSecs += DeltaT              # Note Bene  could advance fraction of time window here
        else:
            startSecs = 0    # if file was opened in getSamples, then startSecs should be set there
        priorPtr = wavFileStartPtr
        if wavFileStartPtr == len(wavfilelist):
            alldone = True
print("ALL DONE WITH:")
print('Database keys', h5db.keys())
for key in h5db.keys():
    print(h5db[key])
    for k in h5db[key].attrs.keys():
            print(f"attribute: {k} => {h5db[key].attrs[k]}")
    for key2 in h5db[key].keys():
        print("    ", h5db[key][key2])
        for k in h5db[key][key2].attrs.keys():
            print(f"          attribute: {k} => {h5db[key].attrs[k]}")

print("h5db file is ", h5SaveDir + h5filename)
h5db.close()
tstop = time.time()
print("To calculate and encode {} spectrograms, the Elapsed time is s {}, m {}, hr {:.2f} encodings/s {:.2f} "\
      .format(totRecords, int(tstop - tstart),\
 int((tstop - tstart) / 60.0),((tstop - tstart) / 3600.0), totRecords/(tstop - tstart) ))
