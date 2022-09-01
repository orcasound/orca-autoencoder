#!/usr/bin/python3
from tensorflow import keras
import h5py
import helpers as h
import time
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import random
import helpers as d3
#######################################
def getArraysFromWavH5(h5db, evalFrac):
    x_train = []
    x_train_lbl = []
    x_eval = []
    x_eval_lbl = []
    for wav in h5db.keys():
        encArrays = h5db['{}/encoders'.format(wav)]
        for enc in encArrays:
            if enc[1] != -1:
                f = random.random()
                if f > evalFrac:
                    x_train.append(enc[4])
                    x_train_lbl.append(enc[1])
                else:
                    x_eval.append(enc[4])
                    x_eval_lbl.append(enc[1])
    return np.asarray(x_train), np.asarray(x_train_lbl), np.asarray(x_eval), np.asarray(x_eval_lbl)
#######################################
h5Labeled = "/home/val/PyCharmFiles/TF_val/AEinput_8_24/OS_09_12_21SpecsNormedSquared_11_wavs.h5"
h5Labeled = "../h5files/OS_09_12_21SpecsNormedSquared_1_wavs.h5"

thisModel = "model_4"

fit_history_plot_path = d3.checkDir("../outputFiles/fit_history/")
modelLoadFilename = ""   # start with random weights in the layers
modelLoadFilename = "../models/model_4_1wav_epochs:100:lr0.001"

eval_frac = 0.05    # fraction of train samples to use for eval during fitting
lr = 0.001          # learning rate for fit
Nepochs_0 = 100
Nepochs_addl = 200
save_ModelDir = d3.checkDir("../models/{}_1wav_epochs:{}:lr{}".format(thisModel, Nepochs_0+Nepochs_addl, lr))


h5db = h5py.File(h5Labeled, 'r+')
print(h5Labeled, ' Database keys \n', h5db.keys())
# for k in h5db.attrs.keys():
#     print(f"     attribute: {k} => {h5db.attrs[k]}")


# use these annotated encoder spectrograms to build train and test datasets.
#  start with train and use a fraction for the evaluation portion.
x_train, x_train_lbl, x_eval, x_eval_lbl = getArraysFromWavH5(h5db, eval_frac)


if modelLoadFilename == "":
    if "model_2" in thisModel:
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[16, 16, 8]),
            keras.layers.Dense(300, activation="relu"),
            keras.layers.Dropout(.2),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
    ])
    if "model_3" in thisModel:
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[16, 16, 8]),
            keras.layers.Dense(400, activation="relu"),
            keras.layers.Dropout(.2),
            keras.layers.Dense(200, activation="relu"),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
    if "model_4" in thisModel:
        model = keras.models.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(16, 16, 8)),
            keras.layers.Dropout(.2),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),  # haves the dimensions

            keras.layers.Conv2D(32, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            # # The third convolution
            # keras.layers.Conv2D(64, (3,3), activation='relu'),
            # keras.layers.MaxPooling2D(2,2),# The fourth convolution
            # keras.layers.Conv2D(64, (3,3), activation='relu'),
            # keras.layers.MaxPooling2D(2,2),# # The fifth convolution
            # keras.layers.Conv2D(64, (3,3), activation='relu'),
            # keras.layers.MaxPooling2D(2,2),# Flatten the results to feed into a DNN
            keras.layers.Flatten(),# 512 neuron hidden layer
            keras.layers.Dense(512, activation='relu'),# Only 1 output neuron. \
            # It will contain a value from 0-1 where 0 for 1 class ('backgnd') and 1 for the other ('call')
            keras.layers.Dense(1, activation='sigmoid')
        ])
else:
    model = keras.models.load_model(modelLoadFilename)

print(thisModel, model.summary())

# ##############################
# from sklearn.metrics import roc_auc_score
# from keras.callbacks import Callback


#
# class RocCallback(Callback):
#     def __init__(self,training_data,validation_data):
#         self.x = training_data[0]
#         self.y = training_data[1]
#         self.x_val = validation_data[0]
#         self.y_val = validation_data[1]
#
#
#     def on_train_begin(self, logs={}):
#         return
#
#     def on_train_end(self, logs={}):
#         return
#
#     def on_epoch_begin(self, epoch, logs={}):
#         return
#
#     def on_epoch_end(self, epoch, logs={}):
#         y_pred_train = self.model.predict(self.x)
#         roc_train = roc_auc_score(self.y, y_pred_train)
#         y_pred_val = self.model.predict(self.x_val)
#         roc_val = roc_auc_score(self.y_val, y_pred_val)
#         print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
#
#         return
#
#     def on_batch_begin(self, batch, logs={}):
#         return
#
#     def on_batch_end(self, batch, logs={}):
#         return
#
# roc = RocCallback(training_data=(x_train, x_train_lbl),
#                   validation_data=(x_eval, x_eval_lbl))
# #############################

#
# def plotPredictionHistograms(thisModel, model, epochs, x_train, x_test, x_eval):
#     pred1 = model.predict(x_train)
#     pred2 = model.predict(x_test)
#     pred3 = model.predict(x_eval)
#     fig, ax = plt.subplots(3, 1, figsize = (8,15))
#     fig.suptitle("Histograms of predictions:\n {} at {} epochs".format(thisModel, epochs), fontsize=32)
#     ax[0].hist(pred1, bins=20)
#     ax[0].set_title('x_train', fontsize=32)
#     ax[1].hist(pred2, bins=20)
#     ax[1].set_title('x_test', fontsize=32)
#     ax[2].hist(pred3, bins=20)
#     ax[2].set_title('x_eval', fontsize=32)
#     plt.tight_layout()
#     plt.show()
#
#
# plotPredictionHistograms(thisModel, model, Nepochs_0, x_train, x_test, x_eval)
# h.printConfusionMatrix(model, 'x_train', x_train, y_train)
# h.printConfusionMatrix(model, 'x_test', x_test, y_test)
# h.printConfusionMatrix(model, 'x_eval', x_eval, y_eval)
#
# #############################

model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=lr), metrics=["accuracy"])
print(model.summary)
tstart = time.time()
history = model.fit(x_train, x_train_lbl, epochs=Nepochs_addl) #, validation_split= 0.1,  shuffle=False)   #, callbacks=[roc])
tstop = time.time()

print('call model evaluate evalspecs')
score = model.evaluate(x_eval, x_eval_lbl)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#
filename = "_history_epochs_{}_lr{}.jpg".format(thisModel, Nepochs_0+Nepochs_addl, lr)
h.show_history(thisModel, history, len(x_train), score, Nepochs_0+Nepochs_addl, fit_history_plot_path, filename)
print("Plot history at ", fit_history_plot_path, filename)
print("\nRun NN for {} epochs, the Elapsed time is s {}, m {}, hr {:.2f} sec/epoch {:.2f} ".format(Nepochs_addl, int(tstop - tstart),\
 int((tstop - tstart) / 60.0),((tstop - tstart) / 3600.0), (tstop - tstart)/Nepochs_addl ))
print("\nSave model at: ", save_ModelDir, "\n")
model.save(save_ModelDir)

h.printConfusionMatrix(model, 'x_train', x_train, x_train_lbl)
#h.printConfusionMatrix(model, 'x_test', x_test, y_test)
h.printConfusionMatrix(model, 'x_eval', x_eval, x_eval_lbl)
#
#plotPredictionHistograms(thisModel, model, Nepochs_0+Nepochs_addl, x_train,  x_eval)
#
# # update database with new predictions
# pred1 = model.predict(x_train)
# pred2 = model.predict(x_test)
# pred3 = model.predict(x_eval)
#
# labels = h5db['train_encs_lbls']
# newlabels = np.zeros([len(pred1), 3])
# for i in range(len(pred1)):
#     newlabels[i][0] = h5db['train_encs_lbls'][i][0]
#     if pred1[i][0] > 0.5:
#         lbl = 1
#     else:
#         lbl = 0
#     newlabels[i][1] = lbl
#     newlabels[i][2] = pred1[i][0]
#     # if i>3976:
#     #     print(newlabels[i][2], pred1[i][0])
# labels[...] = newlabels
#
# labels = h5db['test_encs_lbls']
# newlabels = np.zeros([len(pred2), 3])
# for i in range(len(pred2)):
#     newlabels[i][0] = h5db['test_encs_lbls'][i][0]
#     if pred2[i][0] > 0.5:
#         lbl = 1
#     else:
#         lbl = 0
#     newlabels[i][1] = lbl
#     newlabels[i][2] = pred2[i][0]
# labels[...] = newlabels
#
# labels = h5db['eval_encs_lbls']
# newlabels = np.zeros([len(pred3), 3])
# for i in range(len(pred3)):
#     newlabels[i][0] = h5db['eval_encs_lbls'][i][0]
#     if pred3[i][0] > 0.5:
#         lbl = 1
#     else:
#         lbl = 0
#     newlabels[i][1] = lbl
#     newlabels[i][2] = pred3[i][0]
# labels[...] = newlabels
# print("Complete train_encs_lbls:", h5db['train_encs_lbls'][1:25])
# h5db.close()
#
# h5db = h5py.File(h5Labeled, 'r')
# print("Check to see if database has been updated with predicitions")
# print("Complete train_encs_lbls:", h5db['train_encs_lbls'][1:25])
# y_train = h5db['train_encs_lbls'][:,0]
# print("Tot num of y_train=", len(y_train), "number of 1's = ", np.sum(y_train))
# h5db.close()
# print("updated data base is saved as:\n", h5Labeled)
