import numpy as np
import matplotlib.pyplot as plt
import os
import math
import h5py
import random
import helpers
from tensorflow import keras
import time
import copy

####################################################################
def getArraysFromWavH5(h5s):
    x_train = []
    x_train_lbl = []
    x_test = []
    x_test_lbl = []
    x_eval = []
    x_eval_lbl = []
    wavFiles = []
    totCnt = 0
    for h5 in h5s:
        encoders = h5['audio']['encodings']
        totCnt += len(encoders)
        Ncall = 0
        Nback = 0
        Nuncatagorized = 0

        for enc in encoders:
            if enc[2] == -1:
                Nuncatagorized += 1
            else:
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

    print("Ncall", Ncall, "Nback", Nback, "Nuncatagorized", Nuncatagorized, "Total", Ncall + Nback + Nuncatagorized, "totCnt)", totCnt)
    print("wavFiles",wavFiles)
    return np.asarray(x_train), np.asarray(x_train_lbl), np.asarray(x_test), np.asarray(x_test_lbl), np.asarray(x_eval), np.asarray(x_eval_lbl), wavFiles


####################################################################

h5in_filename = "../h5files/09_12_21_11_files_1secAdvanceWencoder_train_test_eval_labeled.h5"
h5in_filename = "../h5files/wavsAudacityLabeled_4wavsTHREE.h5"

classifierModelLoadFilename = "../models/classifierModel_4_1wav_epochs:100:lr0.001"
classifierModelLoadFilename = "../models/classifierModel_4_1wav_epochs:2500:lr0.001"

classifierModelLoadFilename = "../models/Classifier_0_1wav_epochs:200:lr0.001"  # the TINY classifier

h5in = h5py.File(h5in_filename, 'r')


x_train, x_train_lbl, x_test, x_test_lbl, x_eval, x_eval_lbl, wavFiles = getArraysFromWavH5([h5in])

fit_history_plot_path = helpers.checkDir("../outputFiles/fit_history/")
lr = helpers.metadata['lr']
Nepochs_0 = 200
Nepochs_addl = 200
thisModel = "Classifier_0"

save_ClassifierModelDir = helpers.checkDir("../models/{}_1wav_epochs:{}:lr{}".format(thisModel, Nepochs_0+Nepochs_addl, lr))

if classifierModelLoadFilename == "":
    if thisModel == "Classifier_0":
        classifierModel  = keras.models.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(16, 16, 8)),
            keras.layers.Dropout(.2),

            keras.layers.MaxPooling2D(2, 2),  # Flatten the results to feed into a DNN
            keras.layers.Flatten(),  # 512 neuron hidden layer
            keras.layers.Dense(32, activation='relu'),  # Only 1 output neuron. \
            # It will contain a value from 0-1 where 0 for 1 class ('backgnd') and 1 for the other ('call')
            keras.layers.Dense(1, activation='sigmoid')
        ])
    if thisModel == "Classifier_1":
        classifierModel  = keras.models.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(16, 16, 8)),
            keras.layers.Dropout(.2),

            keras.layers.MaxPooling2D(2, 2),  # Flatten the results to feed into a DNN
            keras.layers.Flatten(),  # 512 neuron hidden layer
            keras.layers.Dense(64, activation='relu'),  # Only 1 output neuron. \
            # It will contain a value from 0-1 where 0 for 1 class ('backgnd') and 1 for the other ('call')
            keras.layers.Dense(1, activation='sigmoid')
        ])
    if thisModel == "Classifier_2":
        classifierModel  = keras.models.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(16, 16, 8)),
            keras.layers.Dropout(.2),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),  # haves the dimensions
            keras.layers.Conv2D(32, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            # The third convolution
            # keras.layers.Conv2D(64, (3,3), activation='relu'),
            # keras.layers.MaxPooling2D(2,2),# The fourth convolution
            # keras.layers.Conv2D(64, (3,3), activation='relu'),
            # keras.layers.MaxPooling2D(2,2),# # The fifth convolution
            # keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),# Flatten the results to feed into a DNN
            keras.layers.Flatten(),# 512 neuron hidden layer
            keras.layers.Dense(512, activation='relu'),# Only 1 output neuron. \
            # It will contain a value from 0-1 where 0 for 1 class ('backgnd') and 1 for the other ('call')
            keras.layers.Dense(1, activation='sigmoid')
        ])
else:
    classifierModel = keras.models.load_model(classifierModelLoadFilename)

print(classifierModel.summary())
classifierModel.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=lr), metrics=["accuracy"])


# run previously trained classifier on some encoded specs
print("/nPredictions on training encodings PRIOR to this training session")
predictions = classifierModel.predict(np.asarray(x_train))
TP, FP, FN, TN, precision, recall = helpers.evaluateClassifier(predictions, x_train_lbl)
print("TP", TP, "FP", FP)
print("FN", FN, "TN", TN)
print("precision {:0.2f}".format(precision))
print("recall    {:0.2f}".format(recall))




tstart = time.time()
history = classifierModel.fit(x_train, x_train_lbl, epochs=Nepochs_addl, validation_data=(x_test, x_test_lbl)) #, validation_split= 0.1,  shuffle=False)   #, callbacks=[roc])
tstop = time.time()

print("Classsifier: Number of epochs {} Learning rate {} Elapsed time is s {}, m {}, hr {:.2f} encodings/s {:.2f} "\
      .format(Nepochs_addl, lr, (tstop - tstart),\
 int((tstop - tstart) / 60.0),((tstop - tstart) / 3600.0), Nepochs_addl/(tstop - tstart) ))

print('call classifierModel evaluate evalspecs')
score = classifierModel.evaluate(x_eval, x_eval_lbl)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#
filename = "_history_epochs_{}_lr{}.jpg".format(thisModel, Nepochs_0+Nepochs_addl, lr)
helpers.show_history(thisModel, history, len(x_train), score, Nepochs_0+Nepochs_addl, fit_history_plot_path, filename)
print("Plot history at ", fit_history_plot_path, filename)
print("\nRun NN for {} epochs, the Elapsed time is s {}, m {}, hr {:.2f} sec/epoch {:.2f} ".format(Nepochs_addl, int(tstop - tstart),\
 int((tstop - tstart) / 60.0),((tstop - tstart) / 3600.0), (tstop - tstart)/Nepochs_addl ))
print("\nSave classifierModel at: ", save_ClassifierModelDir, "\n")
classifierModel.save(save_ClassifierModelDir)

helpers.printConfusionMatrix(classifierModel, 'x_train', x_train, x_train_lbl)
helpers.printConfusionMatrix(classifierModel, 'x_test', x_test, x_test_lbl)
helpers.printConfusionMatrix(classifierModel, 'x_eval', x_eval, x_eval_lbl)

print("\nPredictions post training")
print("/nPredictions prior to training")
predictions = classifierModel.predict(np.asarray(x_train))
TP, FP, FN, TN, precision, recall = helpers.evaluateClassifier(predictions, x_train_lbl)
print("TP", TP, "FP", FP)
print("FN", FN, "TN", TN)
print("precision {:0.2f}".format(precision))
print("recall    {:0.2f}".format(recall))

h5in.close()
