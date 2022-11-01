import numpy as np
import matplotlib.pyplot as plt
import os
import math
import h5py
import random
import helpers
from tensorflow import keras
import time
import pickle
###################################################################
spec_datatype = np.dtype([('wav', '<S30'), ('idx', '<i8'), ('lbl', '<i8'), ('lblp', '<i8'), ('parm1', '<f8'), ('ary', '<f8', (256, 256))])
enc_datatype =  np.dtype([('wav', '<S30'), ('idx', '<i8'), ('lbl', '<i8'), ('lblp', '<i8'), ('parm1', '<f8'),  ('ary', '<f8', (16, 16, 8))])

####################################################################
h5in_1filename = "../h5files/wavsAudacityLabeled_4wavsNNencodings_2.h5"
h5newEncodingsFile = "../h5files/wavsAudacityLabeled_4wavsNNencodings_3.h5"

h5in_1filename = "../h5files/wavsAudacityLabeled_4wavsTHREE.h5"
h5newEncodingsFile = "../h5files/wavsAudacityLabeled_4wavsTHREE_CNNlabeled_2.h5"
classifierModelLoadFilename = "../models/Classifier_0_1wav_epochs:400:lr0.001"


h5in_1 = h5py.File(h5in_1filename, 'r')   # Keys are ['audio']  <KeysViewHDF5 ['encodings', 'spectrograms']>
encs = h5in_1['audio']['encodings']
specs = h5in_1['audio']['spectrograms']
# encs = h5in_1['train_encoders']
# specs = h5in_1['train_specs']

lr = helpers.metadata['lr']
classifierModel = keras.models.load_model(classifierModelLoadFilename)
classifierModel.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=lr), metrics=["accuracy"])
print(classifierModel.summary())

encodings = []
exlabels  = []
for enc in encs:
    encodings.append(enc[5])
    exlabels.append(enc[2])

predictions = classifierModel.predict(np.asarray(encodings))
# print("PRIOR TO PREDICTING")
# TP, FP, FN, TN, precision, recall = helpers.evaluateClassifier(predictions, exlabels)
# print("TP", TP, "FP", FP)
# print("FN", FN, "TN", TN)
# print("precision {:0.2f}".format(precision))
# print("recall    {:0.2f}\n".format(recall))

changeUnclassedIndices = {}
changeIndices = {}
changeClassifiedIndices = {}

lblOneCnt = 0
lblZeroCnt = 0
lblMinusCnt = 0
for i in range(len(encodings)):
    pred = predictions[i][0]
    lbl  = exlabels[i]
    if lbl == 1: lblOneCnt += 1
    if lbl == 0: lblZeroCnt += 1
    if lbl == -1: lblMinusCnt += 1
    encRecord = encs[i]
    specRecord = specs[i]
    if pred > 0.90:
        # use NN to label this a call
        predLbl = 1
        changeIndices[i] = (predLbl, pred, encRecord[0], encRecord[1])
        if lbl != -1:
            if lbl != predLbl:
                changeClassifiedIndices[i] = (predLbl, pred, lbl, encRecord[0], encRecord[1])
        else:
            changeUnclassedIndices[i] = (predLbl, pred, encRecord[0], encRecord[1])

    if pred < 0.1:
        # use NN to label this as NOT a call
        predLbl = 0
        changeIndices[i] = (predLbl, pred, encRecord[0], encRecord[1])
        if lbl != -1:
            if lbl != predLbl:
                changeClassifiedIndices[i] = (predLbl, pred, lbl, encRecord[0], encRecord[1])
        else:
            changeUnclassedIndices[i] = (predLbl, pred, encRecord[0], encRecord[1])
print("\nNumber labeled as:  Calls={},  Not a call={}, Not Labeled={}".format(lblOneCnt, lblZeroCnt, lblMinusCnt))
print("Num unclassed to get predicted labels", len(changeUnclassedIndices))
print("total num with prediction above HIGH or below LOW thresholds",len(changeIndices),"of", len(encs))
print("Num passing thresholds with prediction at odds with label", len(changeClassifiedIndices),"of",len(specs))
# for i in range(len(changeUnclassedIndices)):
#     if i in changeUnclassedIndices.keys():
#         print(i, changeUnclassedIndices[i])

a_file = open("../outputFiles/changeDicts/changeUnClassifiedIndices.pkl", "wb")
pickle.dump(changeUnclassedIndices, a_file)
a_file.close()

specsChg = []
encsChg  = []
lblsChg  = []

specObjs = []
encObjs  = []
cngCnt = 0
iChg = []
#goodIdxs = [44,63,65,70,92,180,281,338,469,472,496,497,668,1048,1049]
for i in range(len(encs)):
    enc = encs[i]
    spec = specs[i]
    if i in changeUnclassedIndices.keys():
        #if i in goodIdxs:
        # wav = enc[0]
        # index = end[1]
        # lbl = enc[2]
        # predLbl = enc[3]
        # predFrac = enc[4]
        # enc = enc[5]
        preds = changeUnclassedIndices[i]
        enc[2] = preds[0]   ##############  HERE WE ARE CHANGING THE MAIN LABEL
        enc[3] = preds[0]
        enc[4] = -preds[1]  ##########   MINUS SIGNIFIES THAT NN JUST CLASSIFIED THIS RECORD
        spec[2] = preds[0]   ##############  HERE WE ARE CHANGING THE MAIN LABEL
        spec[3] = preds[0]
        spec[4] = -preds[1]
        specsChg.append(spec)
        encsChg.append(enc[5])
        lblsChg.append(enc[2])
        cngCnt += 1
        iChg.append(i)
    else:
        enc[4] = predictions[i][0]  # put the NN prediction fraction in the record

    specObjs.append(spec)
    encObjs.append(enc)
print("Number of originally unclassed records changed is ", cngCnt)

thekey = input("Type y to save h5 file with these changes")
if thekey == 'y':
    h5new = h5py.File(h5newEncodingsFile, 'w')
    h5new.create_group('audio')
    h5new['audio'].create_dataset('spectrograms', data=specObjs, maxshape=(None,), dtype=spec_datatype)
    h5new['audio'].create_dataset('encodings', data=encObjs, maxshape=(None,), dtype=enc_datatype)
    print("New spectrograms/encodings/labels are stored in", h5newEncodingsFile)
    h5new.flush()

# h5in_1 = h5py.File(h5in_1filename, 'r')
# h5in_1.create_group('audio')
encs = h5new['audio']['encodings']
specs = h5new['audio']['spectrograms']
encodings = []
exlabels  = []
for enc in encs:
    encodings.append(enc[5])
    exlabels.append(enc[2])

predictions = classifierModel.predict(np.asarray(encodings))
print("AFTER SELECTED PREDICTIONS")
TP, FP, FN, TN, precision, recall = helpers.evaluateClassifier(predictions, exlabels)
print("TP", TP, "FP", FP)
print("FN", FN, "TN", TN)
print("precision {:0.2f}".format(precision))
print("recall    {:0.2f}\n".format(recall))

h5new.close()

print("New encodings file", h5newEncodingsFile)




