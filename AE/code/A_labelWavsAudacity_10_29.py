import numpy as np
import matplotlib.pyplot as plt
import os
import math
import h5py
import random
import helpers
import time
import copy
import soundfile as sf
from tensorflow import keras
"""

Read in Audacity labels
Find corresponding wav file
Read in wav file and 
    Calculate spectrograms
    Use Autoencoder to calculate encoded spectrograms
    Write spectrograms, encoded spectrograms, and labels (filename, lbl, -1, -1.0)   # last two are placeholder for
            binary classifier label and corresponding prediction floating point number
            
"""


Tspec = helpers.metadata['Tspec']    # seconds in spectrogram
DeltaT = helpers.metadata['DeltaT']  # time shift for each spectrogram
f_low = helpers.metadata['f_low']
f_high = helpers. metadata['f_high']
Ntimes = helpers.metadata['Ntimes']  #  dimensions of output spectrograms
Nfreqs = helpers.metadata['Nfreqs']
Nfft = helpers.metadata['Nfft']
logscale = helpers.metadata['logscale']

def calculateSpectrogram(WAV, startsecs):
    Tspec = helpers.metadata['Tspec']  # seconds in spectrogram
    DeltaT = helpers.metadata['DeltaT']
    Ntimes = helpers.metadata['Ntimes']
    f_low = helpers.metadata['f_low']
    f_high = helpers.metadata['f_high']
    Nfreqs = helpers.metadata['Nfreqs']
    logscale = helpers.metadata['logscale']

    with sf.SoundFile(WAV) as f:
        blocksize = int(Tspec * f.samplerate // Ntimes)  # this is one bin in the final spectrogram
        samplerate = f.samplerate

    samples, secsInWav= getSamples(startsecs, int(Tspec * f.samplerate), WAV)

    spectrogram, specCompressed = getCompressedSpectrogram(Ntimes, Nfreqs, f_low, f_high, logscale, samplerate, samples)

    return spectrogram, specCompressed, samplerate, secsInWav

def getSamples(startsecs, Nsamples, WAV):
    # need to get Ntimes blocks of time series data
    channelchoice = -1    # pick channel with higher amplitude
    typedict = {}
    typedict['FLOAT'] = 'float32'
    typedict['PCM_16'] = 'int16'

    NsamplesNeeded = Nsamples
    npsamples = []
    while NsamplesNeeded > 0:

        with sf.SoundFile(WAV) as f:
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
                try:
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
                except Exception as e:
                    print("in get samples", e)
                if len(npsamples) == 0:
                    npsamples = npdata
                else:
                    npsamples = np.append(npsamples, npdata)
            totalSecs = f.seek(0, sf.SEEK_END)/f.samplerate
            f.close()

#    print("n samples", len(npsamples))
    return npsamples, totalSecs

def getCompressedSpectrogram(Ntimes, Nfreqs, f_low, f_high, logscale, samplerate,
                      samples):  # specBlock will be 1-D spectrograms, one for each slice in time
    specGram = []
    samplesPerBin = len(samples) // Ntimes
    for i in range(Ntimes):
        try:
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

            spec = helpers.compressPsdSliceLog(f_values, spec, f_low, f_high, Nfreqs, logscale)
            # if i == 70 or i == 1:
            #     plt.plot(np.log10(spec))
            #     plt.show()
            #     plt.close()
            specGram.append(spec)  # flip to put low frequencies at 'bottom' of array as displayed on screen
            # if i%32 == 0:
            #     print("i=",i)
        except Exception as e:
            print("error in compress spec", e)

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
    # if len(ary[0:nrows - 1, :]) == 0:
    #     print(nrows, len(ary[0:nrows - 1, :]))
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
            try:
                ch0 = np.average(np.abs(np.frombuffer(data, dtype=typedict[f.subtype])[0::2]))
                ch1 = np.average(np.abs(np.frombuffer(data, dtype=typedict[f.subtype])[1::2]))
                if ch0 > ch1:
                    channelchoice = 0
                else:
                    channelchoice = 1
            except:
                channelchoice = 0
        npdata = np.frombuffer(data, dtype=typedict[f.subtype])[channelchoice::2]
    else:
        npdata = np.frombuffer(data, dtype=typedict[f.subtype])
    return npdata

def getLabel(startSecs, lbls):
    for nextIdx in range(len(lbls)):
        labelSecs = float(lbls[nextIdx][0])
        if startSecs < labelSecs:
            break
    if nextIdx == len(lbls):
        return -1  # we are past the last call   Note Bene:  Should look for 0's here but....
    if startSecs > labelSecs - 3:  # within 3 sec of call center, we have a call
        return 1
    if nextIdx == 0 and startSecs > labelSecs - 7:  # within 7 sec of next call we have NO LABEL
        return -1
    if nextIdx > 0:
        priorLabelSecs = float(lbls[nextIdx-1][0])
        if (startSecs > labelSecs - 7 or startSecs < priorLabelSecs + 7):  # within 7 sec of prior or next call we have NO LABEL
            return -1
    return 0   # this is BACKGROUND


def getCorrespondingWav(wav, wavfilelist):
    targetDatetime = helpers.decodeWAVdatetimeFilename(wav)
    for wavfile in wavfilelist:
        if targetDatetime == helpers.decodeWAVdatetimeFilename(wavfile):
            return wavfile
    return None

spec_enc_datatype = np.dtype(
    [('wav', '<S30'), ('idx', '<i8'), ('lbl', '<i8'), ('lblp9', '<i8'), ('parm9', '<f8'), ('lblp1', '<i8'), ('parm1', '<f8'), ('arySpec', '<f8', (256, 256)), ('aryEnc', '<f8', (16, 16, 8))])
enc_datatype = np.dtype(
    [('wav', '<S30'), ('idx', '<i8'), ('lbl', '<i8'), ('lblp9', '<i8'), ('parm9', '<f8'), ('lblp1', '<i8'), ('parm1', '<f8'), ('aryEnc', '<f8', (16, 16, 8))])
spec_datatype = np.dtype(
    [('wav', '<S30'), ('idx', '<i8'), ('lbl', '<i8'), ('lblp9', '<i8'), ('parm9', '<f8'), ('lblp1', '<i8'), ('parm1', '<f8'), ('arySpec', '<f8', (256, 256))])

spec_datatype = np.dtype([('wav', '<S30'), ('idx', '<i8'), ('lbl', '<i8'), ('lblp', '<i8'), ('parm1', '<f8'), ('ary', '<f8', (256, 256))])
enc_datatype =  np.dtype([('wav', '<S30'), ('idx', '<i8'), ('lbl', '<i8'), ('lblp', '<i8'), ('parm1', '<f8'),  ('ary', '<f8', (16, 16, 8))])

################################################################
################################################################
#################################################################
audacityLabelDir = "../labelFiles/audacityLabels/"
h5outFilename = "../h5files/wavsAudacityLabeled_4wavsTHREE.h5"

wavDir = helpers.checkDir("../wavFiles")   # make sure final / is there
useEncoder = True
useClassifier = True

#################################################################
if useClassifier:
    classifierModelLoadFilename = "../models/Classifier_0_1wav_epochs:200:lr0.001"  # the TINY classifier
    classifierModel = keras.models.load_model(classifierModelLoadFilename)
    print(classifierModelLoadFilename)
    print(classifierModel.summary())
    lr = helpers.metadata['lr']
    classifierModel.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=lr),
                        metrics=["accuracy"])

if os.path.exists(h5outFilename):
    os.remove(h5outFilename)
h5out = h5py.File(h5outFilename, mode='a')

homeDir = os.getcwd()
audacityfilelist = helpers.getTxts(audacityLabelDir)

print("Labelling files are:", audacityfilelist)

wavLabels = []
for audacityLabelFile in audacityfilelist:
    wavFile = audacityLabelFile.split("/")[-1].split(".")[0]

    lblfile = open(audacityLabelFile, encoding = 'utf-8')  # these labels are time at center of call and Character for type
        # note -- one Audacity label file corresponds to one wav file.
    lbl_list = []
    for line in lblfile:
        items = line.split("\t")
        if items[0] != "\\":
            lbl_list.append((items[0], items[2][:-1]))
#    print(audacityLabelFile, " List of", wavFile, "labels is:\n", lbl_list)    #  [('10.036570', 'C'), ('26.804985', 'C'), .....
    wavLabels.append([wavFile, lbl_list])
os.chdir(homeDir)  # return to home directory
wavfilelist = helpers.getWavs(wavDir)
os.chdir(homeDir)  # return to home directory
#print("wavLabels loaded", wavLabels)

#  encoder
AEmodelFilenameWts = ""
AEmodelClass = ""
if useEncoder:
    AEmodelFilenameWts =  "../outputFiles/AE_weights/conv_ae_test_10_min_9_18h5_NN_6_epochs_2110_3110"
    AEmodelClass = "AE_6()"
    conv_ae = helpers.AE_6()
    conv_ae.load_weights(AEmodelFilenameWts)

firstH5 = True
recordlist = []
cnt = 0
for wav, lbls in wavLabels:  #OS_9_12_2021_09_07_00.wav [('10.036570', 'C'), ('26.804985', 'C'), ('34.932159', 'C'), ...
    cnt += 1
    print(wav,lbls)
    thisWav = getCorrespondingWav(wav, wavfilelist)
    if thisWav != None:
        doneWithWav = False
        startSecs = 0
        speclist = []
        idxlist = []
        lbllist = []
        while not doneWithWav:
            try:
                if startSecs % 10 == 0:
                    print("Processing wav",thisWav," at ", startSecs,"secs")
                spectrogram, spectrogramNormed, samplerate, secsInWav = calculateSpectrogram(wavDir+thisWav, startSecs)
                if spectrogramNormed.shape == (256, 256):
                    speclist.append(spectrogramNormed)
                    idxlist.append(startSecs * samplerate)
                    lbllist.append(getLabel(startSecs, lbls))
                    startSecs += DeltaT
                    if startSecs >2:
                        doneWithWav = True
                    if startSecs >= secsInWav - 3*DeltaT:
                        doneWithWav = True
            except Exception as e:
                print("got error in spectrogram", e)

        encodedlist = []
        if useEncoder:
            slist = np.expand_dims(speclist, axis=-1)
            encodedlist = conv_ae.encoder.predict(slist)   # use the Autoencoder to calculate the encoding layer
        predictlist = []
        if useClassifier:
            predictlist = classifierModel.predict(np.asarray(encodedlist))
        # Now create the label tuples
        specObjs = []
        encObjs = []
        for i in range(len(speclist)):
            if len(predictlist)>0:
                specObjs.append((thisWav, idxlist[i], lbllist[i], -1, -1.0, speclist[i]))
                encObjs.append((thisWav, idxlist[i], lbllist[i], -1, -1.0, encodedlist[i]))
            else:
                specObjs.append((thisWav, idxlist[i], lbllist[i], -1, predictlist[i], speclist[i]))
                encObjs.append((thisWav, idxlist[i], lbllist[i], -1, predictlist[i], encodedlist[i]))
        if firstH5:
            firstH5 = False
            h5out.create_group('audio')
            h5out['audio'].create_dataset('spectrograms', data=specObjs, maxshape=(None, ), dtype=spec_datatype)
            h5out['audio'].create_dataset('encodings', data=encObjs, maxshape=(None,), dtype=enc_datatype)
#            h5out.create_dataset('specs', data=speclist, compression="gzip", chunks=True, maxshape=(None, ), dtype=np.dtype([('ary', '<f8', (256, 256))]))
        else:
            newLength = h5out['audio']['spectrograms'].shape[0] + len(specObjs)
            h5out['audio']['spectrograms'].resize(newLength, axis=0)
            h5out['audio']['spectrograms'][-len(specObjs):] = specObjs
            newLength = h5out['audio']['encodings'].shape[0] + len(encObjs)
            h5out['audio']['encodings'].resize(newLength, axis=0)
            h5out['audio']['encodings'][-len(encObjs):] = encObjs
        h5out.flush()
    print("finished", thisWav,"Length",len(h5out['audio']['encodings']))
for key in h5out.keys():
    print(key, h5out[key])
    for k in h5out[key].keys():
        print(k, len(h5out[key][k]), "records")

h5out.close()
print("h5out done")

