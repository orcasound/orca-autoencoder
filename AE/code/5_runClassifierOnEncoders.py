import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math
import h5py
import random
import helpers as d3
import time
from tensorflow import keras
"""
  Access h5 database of spectrograms
  Apply trained classifier to each Encoded array
       and write classification back out to the database
  
"""

########################################################################
np.random.seed(42)

h5Inputfilename = "../h5files/OS_09_12_21SpecsNormedSquared_1_wavs.h5"
h5Inputfilename = "../h5files/OS_09_12_21SpecsNormedSquared_11_wavs.h5"

trainedClassifierModel = "../models/model_4_epochs:1500:lr0.001"

model = keras.models.load_model(trainedClassifierModel)

print(model.summary())

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

for wav in h5db.keys():
    thisWavEncs = h5db[wav]['encoders'][:]
    theEncArrays = []
    for enc in thisWavEncs:
        x = np.expand_dims(enc[4], axis=-1)
        theEncArrays.append(x)
        # p = model.predict(np.asarray(theEncArrays))
        # print(p)
    predictions = model.predict(np.asarray(theEncArrays))
    print("finished predicting for h5db", wav)
    for i in range(len(predictions)):
        thisWavEncs[i][3] = predictions[i][0]
        if predictions[i][0] > 0.75:
            thisWavEncs[i][2] = 1
        if predictions[i][0] < 0.25:
            thisWavEncs[i][2] = 0
    del h5db[wav]['encoders']
    h5db[wav]['encoders'] = thisWavEncs

h5db.close()
