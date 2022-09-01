import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math
import h5py
import random
import helpers as h
import time
"""
  Access h5 database of unlabeled calls and read Audacity label file and label these calls
  Labels are updated in the h4 database
"""

########################################################################
np.random.seed(42)
h5Inputfilename = "../h5files/OS_09_12_21SpecsNormedSquared_1_wavs.h5"
h5Inputfilename = "../h5files/OS_09_12_21SpecsNormedSquared_11_wavs.h5"
audacityLabelDir = "../labelFiles/audacityLabels/"


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
parmstrDict = eval(h5db.attrs['paramStr'])  # converts string back to dict
unlabeledCnt = 0
labeledCnt = 0

tstart = time.time()

os.chdir(audacityLabelDir)
txtFiles = glob.glob("*.txt")
buildNewDatabase = True
for audacityLabelFile in txtFiles:
    wavFile = audacityLabelFile.split("/")[-1].split(".")[0]

    lblfile = open(audacityLabelFile, encoding = 'utf-8')  # these labels are time at center of call and Character for type
        # note -- one Audacity label file corresponds to one wav file.
    lbl_list = []
    for line in lblfile:
        items = line.split("\t")
        if items[0] != "\\":
            lbl_list.append((items[0], items[2][:-1]))
    print(audacityLabelFile, " List of labels is:\n", lbl_list)    #  [('10.036570', 'C'), ('26.804985', 'C'), .....
    #first find this wav's database
    thedb = None
    for wav in h5db.keys():
        print(wav, wavFile, h5db[wav].attrs['filename'])
        if wavFile in h5db[wav].attrs['filename']:
            thedb = h5db[wav]
            break

    if thedb:
        print("''''''''''''''''''h5db[wav] ", thedb, "\n wav for labeling is: ", thedb.attrs['filename'])
        tSpecs = 0
        thisWavSpecs = thedb['specs'][:]
        thisWavEncs  = thedb['encoders'][:]
        #    print(thisWavSpecs[0][1])
        #    print(thisWavEncs[0][1])
        specIdx = 0
        labeledSpecs = []
        labeledEncs  = []

        for lbl in lbl_list:
            if specIdx >= len(thisWavSpecs):
                break
            tlblsec = float(lbl[0])
            thislbl = lbl[1]
#            print("this label", lbl, "specIdx", specIdx, thisWavSpecs[specIdx][0], tlblsec, len(thisWavSpecs))
            while thisWavSpecs[specIdx][0] < tlblsec - 3:
                #print(thisWavSpecs[specIdx][0],thisWavSpecs[specIdx][1],thisWavSpecs[specIdx][2])
                thisWavSpecs[specIdx][1] = 0  #  Not a call
                thisWavEncs[specIdx][1]  = 0
                specIdx += 1
                if specIdx >= len(thisWavSpecs):
                    break
                unlabeledCnt += 1
            for i in range(3):
                thisWavSpecs[specIdx][1] = 1  #  These are calls
                thisWavEncs[specIdx][1]  = 1
                specIdx += 1
                if specIdx >= len(thisWavSpecs):
                    break
                labeledCnt += 1
        while specIdx < len(thisWavSpecs):
            thisWavSpecs[specIdx][1] = 0  # Not a call  -- end of wav file
            thisWavEncs[specIdx][1] = 0
            unlabeledCnt += 1
            specIdx += 1

        del thedb['specs']
        del thedb['encoders']
        thedb['specs'] = thisWavSpecs
        thedb['encoders'] = thisWavEncs
        h5db.flush()
        print("updated db for ", wav)

print("\nNumber of background records {}, Number of labeled calls {}".format(unlabeledCnt, labeledCnt))
tstop = time.time()
print("To calculate and encode {} spectrograms, the Elapsed time is s {}, m {}, hr {:.2f} encodings/s {:.2f} ".format(labeledCnt + unlabeledCnt, int(tstop - tstart),\
 int((tstop - tstart) / 60.0),((tstop - tstart) / 3600.0), (labeledCnt + unlabeledCnt)/(tstop - tstart) ))

