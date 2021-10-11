# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import ae_classes_1
import numpy as np
import pickle
import time
import math

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print('device is', device)


def saveModel(model, saveFilename):
    with open(saveFilename, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def loadModel(loadFilename):
    with open(loadFilename, 'rb') as f:
        return pickle.load(f)


def doValidation(jpgFilename):
    # Validation
    with torch.set_grad_enabled(False):
        for validation_set in enumerate(validation_generator):
            #   for validation_set in enumerate(training_generator) :
            # # Transfer to GPU
            # local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            local_batch = validation_set[1]['array']
            local_batch = local_batch.to(device)
            print('len local_batch is', len(local_batch), 'device is ', local_batch.device)
            test_examples = local_batch.view(-1, rows * cols).to(device)
            print('test_examples device', test_examples.device)
            reconstruction = model(test_examples)
            print('reconstruction device is ', reconstruction.device)
            break
    print("len(test_examples)=", len(test_examples))
    print("len(reconstruction)=", len(reconstruction))
    with torch.no_grad():
        ncols = 10
        nrows = 3  # number of rows of PAIRS
        plt.figure(figsize=(20, 20 * (nrows * 2) // ncols))
        index = 0
        for j in range(nrows):
            for i in range(ncols):
                ax = plt.subplot(nrows * 2, ncols, index + i + 1 + j * ncols)
                plt.imshow(test_examples[index + i].detach().cpu().numpy().reshape(rows, cols) + 0.01)
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            for i in range(ncols):
                ax = plt.subplot(nrows * 2, ncols, index + ncols + i + 1 + j * ncols)
                plt.imshow(reconstruction[index + i].detach().cpu().numpy().reshape(rows, cols) + 0.01)
                # plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            index += ncols
        # plt.show()
        plt.savefig(jpgFilename)


# Parameters
params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 6}
validation_fraction = 0.1
init_epochs = 1299
addnl_epochs = 1001
learning_rate = 0.001
thisdate = '10_08'
modelID = 'MLP_4'
loadModelFilename = "../../models/model_AE_4_1299_10_03.pkl"
saveModelFilename = "../../models/model_{}_{}_{}.pkl".format(modelID, init_epochs + addnl_epochs, thisdate)
specPklDir = "../../spectrograms/"
jpgDir = "../../jpgs/"
# Datasets
training_generator, validation_generator, rows, cols = ae_classes_1.setupOrcaDatasets(device, params,
                                                                                      validation_fraction, specPklDir)
# Generators are in my_classes.py
print("generators have ", rows, " and ", cols, "cols")

if loadModelFilename != "":
    model = loadModel(loadModelFilename)
    ##    model.cuda()
    print('************  this existing model ************************')
    print(model)
    print("     model device is ", next(model.parameters()).device)
    print('     try model.eval')
    model.eval()
    print('     try doValidation')
    doValidation(jpgDir + 'model_{}_at_epoch_{}.jpg'.format(modelID, init_epochs))

else:
    model = ae_classes_1.MLP_4(input_shape=rows * cols).to(device)
    print("new model is a MLP:", model)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

print('# mean-squared error loss')
criterion = nn.MSELoss()
print('criterion =', criterion)
# Loop over epochs
totalCnt = 0
tstart = time.time()
for epoch in range(addnl_epochs):
    loss = 0
    cnt = 0
    # Training
    for local_set in training_generator:  # enumerate(training_generator):
        # Transfer to GPU
        local_batch = local_set['array'].to(device)
        local_label = local_set['label'].to(device)
        # Model computations
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        local_batch = local_batch.view(-1, rows * cols).to(device)
        if True in np.isnan(local_batch.tolist()):
            bat = local_batch.tolist()

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        # print('line 139 compute outputs for local_batch on ', local_batch.device)
        outputs = model(local_batch)
        # print('# compute training reconstruction loss', local_batch.device)
        train_loss = criterion(outputs, local_batch)
        # compute accumulated gradients
        train_loss.backward()
        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        cnt += len(local_batch)

    # compute the epoch training loss
    loss = loss / len(local_batch)

    # display the epoch training loss
    if epoch % 10 == 0:
        print("cnts: {} {}, epoch : {}/{}, recon loss = {:.8f}, {:.4f}".format(cnt, totalCnt, epoch + 1, addnl_epochs,
                                                                               loss, math.log10(loss)))
    totalCnt += cnt
    if epoch % 1000 == 0:
        saveModelFilename = "../../models/model_{}_{}_{}.pkl".format(modelID, init_epochs + epoch, thisdate)
        saveModel(model, saveModelFilename)
        tstop = time.time()
        print("Elapsed time (s, m) is ", int(tstop - tstart), int((tstop - tstart) / 60.0), saveModelFilename)
saveModelFilename = "../../models/model_{}_{}_{}.pkl".format(modelID, init_epochs + epoch, thisdate)
saveModel(model, saveModelFilename)
print(model)
model.eval()
doValidation(jpgDir + 'model_{}_at_epoch_{}.jpg'.format(modelID, init_epochs + epoch +1))
tstop = time.time()
print("Elapsed time s {}, m {}, hr {:.2f} s/epoch {:.2f} ".format(int(tstop - tstart), int((tstop - tstart) / 60.0), ((tstop - tstart) / 3600.0),
    (tstop - tstart) / (init_epochs + epoch)))
print("Save model as: ", saveModelFilename)
