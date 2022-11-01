import numpy as np
import random
import h5py
import helpers
from tensorflow import keras
import time


def getArraysFromWavH5(h5s):
    x_train = []
    x_train_lbl = []
    x_test = []
    x_test_lbl = []
    x_eval = []
    x_eval_lbl = []
    wavFiles = []
    for h5 in h5s:
        encoders = h5['audio']['encodings']
        Ncall = 0
        Nback = 0
        Nuncatagorized = 0

        for enc in encoders:
            if enc[2] == -1:
                Nuncatagorized += 1
            if enc[2] == 1:
                Ncall += 1
            if enc[2] == 0:
                Nback += 1
            if enc[0] not in wavFiles:
                wavFiles.append(enc[0])
            f = random.random()
            if f <0.75:
                x_train.append(enc[5])
                x_train_lbl.append(enc[2])
            else:
                if f < 0.9:
                    x_test.append(enc[5])
                    x_test_lbl.append(enc[2])
                else:
                    x_eval.append(enc[5])
                    x_eval_lbl.append(enc[2])

    print("Ncall", Ncall, "Nback", Nback, "Nuncatagorized", Nuncatagorized, "Total", Ncall + Nback + Nuncatagorized)
    print(wavFiles)
    return np.asarray(x_train), np.asarray(x_train_lbl), np.asarray(x_test), np.asarray(x_test_lbl), np.asarray(x_eval), np.asarray(x_eval_lbl), wavFiles


#################################################################

h5in_1filename = "../h5files/wavsAudacityLabeled_4wavs.h5"
h5in_1filename = "../h5files/wavsAudacityLabeled_4wavsNNencodings_1.h5"

h5in_1 = h5py.File(h5in_1filename, 'r')

classifierModelLoadFilename = "../models/classifierModel_4_1wav_epochs:2500:lr0.001"
lr = helpers.metadata['lr']
classifierModel = keras.models.load_model(classifierModelLoadFilename)
classifierModel.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=lr), metrics=["accuracy"])
print(classifierModel.summary)



x_train, x_train_lbl, x_test, x_test_lbl, x_eval, x_eval_lbl, wavFiles = getArraysFromWavH5([h5in_1])

# run previously trained classifier on some encoded specs

predictions = classifierModel.predict(np.asarray(x_train))
# x_train_lbl is list of spectrograms labels: 1's and 0's
# predictions are the predicted values
TP, FP, FN, TN, precision, recall = helpers.evaluateClassifier(predictions, x_train_lbl)

print("TP", TP, "FP", FP)
print("FN", FN, "TN", TN)
print("precision {:0.2f}".format(precision))
print("recall    {:0.2f}".format(recall))

print("The input h5 file is ", h5in_1filename)

h5in_1.close()


