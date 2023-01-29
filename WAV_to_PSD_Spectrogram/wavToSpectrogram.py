import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import copy


def calculateSpectrogram(WAV, startsecs):

    with sf.SoundFile(WAV) as f:
        blocksize = int(Tspec * f.samplerate // Ntimes)  # this is one bin in the final spectrogram
        samplerate = f.samplerate

    samples, secsInWav = getSamples(startsecs, int(Tspec * f.samplerate), WAV)

    spectrogram, specCompressed = getCompressedSpectrogram(Ntimes, Nfreqs, f_low, f_high, logscale, samplerate, samples)

    return spectrogram, specCompressed, samplerate, secsInWav


def getSamples(startsecs, Nsamples, WAV):
    # need to get Ntimes blocks of time series data
    channelchoice = -1  # pick channel with higher amplitude
    typedict = {}
    typedict['FLOAT'] = 'float32'
    typedict['PCM_16'] = 'int16'

    NsamplesNeeded = Nsamples
    npsamples = []
    while NsamplesNeeded > 0:

        with sf.SoundFile(WAV) as f:
            #            print("-------------reading wav file", WAVs[wavStartIdx], "wavStartIdx", wavStartIdx)
            availableSamples = f.seek(0, sf.SEEK_END) - int(startsecs * f.samplerate)
            #            if availableSamples < 0:
            #                print(startsecs)
            #            print("availableSamples=", availableSamples, WAVs[wavStartIdx])
            if len(npsamples) == 0:  # test for first wav file only
                if availableSamples > 0:
                    f.seek(
                        int(startsecs * f.samplerate))  # for first wav file, start at desired number of secs into file
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
                        NsamplesNeeded -= availableSamples
                        startsecs = 0
                        availableSamples = 0
                except Exception as e:
                    print("in get samples", e)
                if len(npsamples) == 0:
                    npsamples = npdata
                else:
                    npsamples = np.append(npsamples, npdata)
            totalSecs = f.seek(0, sf.SEEK_END) / f.samplerate
            f.close()

    #    print("n samples", len(npsamples))
    return npsamples, totalSecs

def setupFreqBands(flow, fhigh, nbands, doLogs):
    df = (fhigh - flow) / nbands
    fbands = np.zeros(nbands+1)  # reserve [0] for the integrated psd (Broadband level)
    if not doLogs:
        for i in range(nbands):
            fbands[i+1] = flow + i*df
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
        while freqs[idxPsd] <= fbands[idxCompressed]:  # step through psd frequencies until greater than this fband
            idxPsd += 1
            inNewBand = True
        deltaf = freqs[idxPsd] - fbands[idxCompressed]  # distance of this psd frequency into this fband
        if DEBUG == 10: print(deltaf)
        if deltaf > dfbands[idxCompressed]:  # have jumped an entire band
            compressedSlice[idxCompressed] += psds[idxPsd]*dfbands[idxCompressed]/df  # frac of psds = slice
            idxCompressed += 1
        else:
            pfrac = deltaf / df
            compressedSlice[idxCompressed] += pfrac * psds[idxPsd]  # put frac of first pwr in psd
            if inNewBand:
                idxCompressed += 1
            idxPsd += 1
        if DEBUG == 10: print(idxPsd, idxCompressed, deltaf, inNewBand)
        if DEBUG == 10: print("")
    compressedSlice[0] = np.sum(compressedSlice)
    return compressedSlice, fbands

def getCompressedSpectrogram(Ntimes, Nfreqs, f_low, f_high, logscale, samplerate,
                             samples):  # specBlock will be 1-D spectrograms, one for each slice in time
    specGram = []
    samplesPerBin = len(samples) // Ntimes
    for i in range(Ntimes):
        try:
            data = samples[i * samplesPerBin: (i + 1) * samplesPerBin]
            data = data * np.hamming(len(data))
            #        data = data * np.blackman(len(data))
            #         plt.plot(data)
            #         plt.show()
            #         plt.close()
            spec = np.abs(np.fft.rfft(data, Nfft))
            f_values = np.fft.fftfreq(Nfft, d=1. / samplerate)
            f_values = f_values[0:Nfft//2+1]  # drop the un-needed negative frequencies
            f_values[-1] = f_values[-2]
            # if i == 70 or i == 1:
            #     plt.plot(np.log10(spec))
            #     plt.show()
            #     plt.close()

            spec, freqs = compressPsdSliceLog(f_values, spec, f_low, f_high, Nfreqs, logscale)
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
    specGram = np.rot90(specGram,
                        3)  ###COULD use  square roots etc to bring lower peaks up  i.e.  0.36  -> 0.6  -> 0.77
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
    bbmax = np.max(ary[nrows - 1, :])  # get max of bottom row
    bbmin = np.min(ary[nrows - 1, :])
    ary[nrows - 1, :] = (ary[nrows - 1, :] - bbmin) / (bbmax - bbmin)  # normalize the bottom row to 0 -> 1
    # if len(ary[0:nrows - 1, :]) == 0:
    #     print(nrows, len(ary[0:nrows - 1, :]))
    aryMean = np.mean(ary[0:nrows - 1, :])
    aryStd = np.std(ary[0:nrows - 1, :])
    ary[0:nrows - 1, :] = (ary[0:nrows - 1, :] - aryMean) / (4 * aryStd)
    ary[ary < -1] = -1
    ary[ary > 1] = 1
    ary = ary / 2.0 + 0.5
    return ary


def convertToNumpy(f, typedict, data):
    channelchoice = -1  # -1 to pick channel with higher amplitude
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

########################################################## RUN starts here

Tspec =3  # seconds in spectrogram
DeltaT = 1  # seconds to advance for next spectrogram
Ntimes = 256
f_low = 100
f_high = 10000
Nfreqs = 256
logscale = False
# Choose "Hamming"  "Blackman"
fftWindow = "Blackman"
fftWindow = "Hamming"
Nfft = 1024

wavDir = ""
thisWav = "210825-1925_L-highlight.wav"

DEBUG = 0

doneWithWav = False
startSecs = 0
while not doneWithWav:
    try:
        if startSecs % 10 == 0:
            print("Processing wav", thisWav, " at ", startSecs, "secs")
        spectrogram, spectrogramNormed, samplerate, secsInWavFile = calculateSpectrogram(wavDir + thisWav, startSecs)
        title = "{:0.2f}s {} Nfreqs {} Ntimes Spectrograms, log = {}".format(startSecs, Nfreqs, Ntimes, logscale)
        fig = plt.figure()
        plt.gray()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_title("Raw spectrogram")
        ax1.imshow(spectrogram)
        ax2.set_title("Normalized then squared")
        ax2.imshow(spectrogramNormed)
        # using padding
        fig.tight_layout(pad=1.0)

        plt.show()
        plt.close()

        input("Time was {} sec. Hit Enter to continue forward in time".format(startSecs))
        startSecs += DeltaT
        if startSecs >= secsInWavFile - DeltaT:
            doneWithWav = True
    except Exception as e:
        print("got error in calculateSpectrogram", e)

