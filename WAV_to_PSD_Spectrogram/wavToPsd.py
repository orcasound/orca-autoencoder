import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

"""
Code that demonstrates converting time series data into power spectral densith (psd)
Parameters that can are selected are given below at the start of the execution

wavDir = "/home/val/PycharmProjects_Original/AEproject/wavFiles/"
thisWav = "DTMF.wav"

with sf.SoundFile(wavDir+thisWav) as f:
    samplerate = f.samplerate
flow = 10
fhigh = samplerate/2  # set high frequency cutoff at the Nyquist frequency
nbands = 33
Nfft = 1024
doLogs = False
# Choose "Hamming"  "Blackman"
fftWindow = "Blackman"
fftWindow = "Hamming"
deltaT = 1  # one second psd's

First deltaT's worth of samples are taken from wav file. (If stereo, the channel with highes signal is used.)
PSD is calculated using numpy's fft routine for real numbers. ( np.abs(np.fft.rfft(data, Nfft)) )
Then the PSD is integrated into the frequency range and bin size(s)  ( If doLogs is True, bin sizes increase logarithmetically.)
Finally, the timeseries, raw psd and compressed-into-frequency bins data are plotted

"""

def convertToNumpy(f, typedict, data, channelchoice):
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

def getSamples(startsecs, Nsamples, WAV):

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

            if len(npsamples) == 0:
                if availableSamples > 0:
                    f.seek(int(startsecs * f.samplerate))  # for first wav file, start at desired number of secs into file
                else:
                    f.seek(0)  # start at beginning of wav file, continuing into a new file
            while availableSamples > 0 and NsamplesNeeded > 0:
                try:
                    if availableSamples >= NsamplesNeeded:
                        data = f.buffer_read(NsamplesNeeded, dtype=typedict[f.subtype])
                        npdata = convertToNumpy(f, typedict, data, channelchoice)
                        NsamplesNeeded = 0
                    else:
                        data = f.buffer_read(availableSamples, dtype=typedict[f.subtype])
                        npdata = convertToNumpy(f, typedict, data, channelchoice)
                        NsamplesNeeded -= availableSamples
                        startsecs = 0
                        availableSamples = 0
                except Exception as e:
                    print("Exception in get samples is", e)
                if len(npsamples) == 0:
                    npsamples = npdata
                else:
                    npsamples = np.append(npsamples, npdata)
            totalSecsInFile = f.seek(0, sf.SEEK_END) / f.samplerate
            f.close()
    #    print("n samples", len(npsamples))
    return npsamples, totalSecsInFile

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

def calculatePSD(wavFile, startSecs, Nsamples, flow, fhigh, nbands, Nfft, doLogs):
    samples, secsIntoWav = getSamples(startSecs, Nsamples, wavFile)

    if fftWindow == "Hamming":  data = samples * np.hamming(len(samples))
    if fftWindow == "Blackman": data = samples * np.blackman(len(samples))

    psd = np.abs(np.fft.rfft(data, Nfft))
    f_values = np.fft.fftfreq(Nfft, d=1. / samplerate)
    f_values = f_values[0:Nfft//2+1]  # drop the un-needed negative frequencies
    f_values[-1] = f_values[-2]

    psdCompressed, fbands = compressPsdSliceLog(f_values, psd, flow, fhigh, nbands, doLogs)
    title = "{:0.2f}s {} Frequency bands, log = {}".format(startSecs, nbands, doLogs)
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.plot(data)
    ax1.set_title("Time series {} window".format(fftWindow))
    ax2.plot(f_values, np.log10(psd),marker='o')
    ax2.set_title("log10(PSD", fontsize = 9)
    ax3.plot(fbands[1:], np.log10(psdCompressed[1:]),marker='o')  # Don't graph the DC broadband level
    ax3.set_title(title)
    # using padding
    fig.tight_layout(pad=1.0)
    plt.show()
    plt.close()
    return psdCompressed, secsIntoWav


################################################################  RUN STARTS HERE

wavDir = ""
thisWav = "someCalls.wav"

with sf.SoundFile(wavDir+thisWav) as f:
    samplerate = f.samplerate
flow = 10
fhigh = samplerate/2  # set high frequency cutoff at the Nyquist frequency
nbands = 256   # 33 gives about 1/3 octave bands from 10hz to 22khz
Nfft = 1024
doLogs = False
# Choose "Hamming"  "Blackman"
fftWindow = "Blackman"
fftWindow = "Hamming"
deltaT = 1  # one second psd's

DEBUG = 0

doneWithWav = False
startSecs = 0
while not doneWithWav:
    try:
        if startSecs % 10 == 0:
            print("Processing wav", thisWav, " at ", startSecs, "secs")
        psd, secsInWavFile = calculatePSD(wavDir + thisWav,  startSecs, deltaT*samplerate, flow, fhigh, nbands, Nfft, doLogs)
        input("Time was {} sec. Hit Enter to continue forward in time".format(startSecs))
        startSecs += deltaT
        if startSecs >= secsInWavFile - deltaT:
            doneWithWav = True
    except Exception as e:
        print("got error in calculatePSD", e)
