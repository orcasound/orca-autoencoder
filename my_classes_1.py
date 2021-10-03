# my classes for pytorch project   https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision   #  for MNIST datasets
import helpers_3D as d3
from random import random
import numpy as np
import os
import pickle
import math

class myDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, device, dataArys, lblAry):
        'Initialization'
        super(myDataset, self).__init__()

        self.array_data = torch.tensor(dataArys, dtype=torch.float32).to(device)
        self.label_data = torch.tensor(lblAry, dtype=torch.long).to(device)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.array_data)

  def __getitem__(self, index):
        'Generates one sample of data'
        if torch.is_tensor(index):
          index = index.tolist()
        return {'array': self.array_data[index], 'label': self.label_data[index]}

def setupMNISTdatasets(batch_size):
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

  train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
  )
  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
  )

  test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
  )
  test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=10, shuffle=False
  )

  return train_loader, test_loader, test_loader.dataset.data.shape[1], test_loader.dataset.data.shape[2]

def setupOrcaDatasets(device, params, validFrac, specPklDir):
  fileList = []
  itms = os.listdir(specPklDir)
  for it in itms:
    if 'pkl' in it and 'Summary' not in it:
      fileList.append(specPklDir + it)
  dblist = []
  lbls = []
  iStart = 0
  for pklfile in fileList:
    print('Reading file ', pklfile)
    specObjs = d3.load_obj(pklfile)

    for ary in specObjs.arrayList:
      if len(ary) > 0:  ######  Note Bene  Only if averages have > zero length
        for ar in ary:
          dblist.append(ar)
#    dblist.append(specObjs.arrayList[0])   ##########  Note Bene  only using 'real-time' spectrograms now

  for i in range(len(dblist)):  ### Note Bene  Give each spec a unique label for now
      lbls.append(i)

  print("Number of spectrograms is ", len(dblist))
  specs_train = []
  lbls_train = []
  specs_valid = []
  lbls_valid = []
  for i in range(len(dblist)):
    if True not in np.isnan(dblist[i].tolist()):  # DO NOT ACCEPT SPECTROGRAMS THAT HAVE nans
      if random() < validFrac:
        specs_valid.append(dblist[i])
        lbls_valid.append(lbls[i])
      else:
        specs_train.append(dblist[i])
        lbls_train.append(lbls[i])
    else:
      print("GOT NANS in array number ", i)
  lbls_valid = torch.tensor(np.array(lbls_valid), dtype=torch.long).to(device)
  lbls_train = torch.tensor(np.array(lbls_train), dtype=torch.long).to(device)
  specs_valid = torch.tensor(np.array(specs_valid), dtype=torch.float32).to(device)
  specs_train = torch.tensor(np.array(specs_train), dtype=torch.float32).to(device)

  training_data = myDataset(device, specs_train, lbls_train)
  training_generator = torch.utils.data.DataLoader(
    training_data, batch_size=params['batch_size'], shuffle=params['shuffle']
  )
  validation_data = myDataset(device, specs_valid, lbls_valid)
  validation_generator = torch.utils.data.DataLoader(
    validation_data, batch_size=params['batch_size'], shuffle=params['shuffle']
  )

  print('in setupOrcaDatasets -- validation generator output') 
  for validation_set in enumerate(validation_generator):
    local_batch = validation_set[1]['array']
    print(local_batch[0],'device is ',local_batch.device)

  return training_generator, validation_generator, dblist[0].shape[0], dblist[0].shape[1]



class AE_4(nn.Module):
  def __init__(self, **kwargs):
    super(AE_4, self).__init__()

    # Encoder
    self.enc1 = nn.Linear(in_features=kwargs["input_shape"], out_features=512)  # Input image (28*28 = 784)
    self.enc2 = nn.Linear(in_features=512, out_features=256)
    self.enc3 = nn.Linear(in_features=256, out_features=128)
    self.enc4 = nn.Linear(in_features=128, out_features=64)
 #   self.enc5 = nn.Linear(in_features=64, out_features=32)
#    self.enc6 = nn.Linear(in_features=32, out_features=16)
    # Decoder
#    self.dec1 = nn.Linear(in_features=16, out_features=32)
#    self.dec2 = nn.Linear(in_features=32, out_features=64)
    self.dec3 = nn.Linear(in_features=64, out_features=128)
    self.dec4 = nn.Linear(in_features=128, out_features=256)
    self.dec5 = nn.Linear(in_features=256, out_features=512)
    self.dec6 = nn.Linear(in_features=512, out_features=kwargs['input_shape'])  # Output image (28*28 = 784)

  def forward(self, x):
    x = F.relu(self.enc1(x))
    x = F.relu(self.enc2(x))
    x = F.relu(self.enc3(x))
    x = F.relu(self.enc4(x))
#    x = F.relu(self.enc5(x))
#    x = F.relu(self.enc6(x))

#    x = F.relu(self.dec1(x))
#    x = F.relu(self.dec2(x))
    x = F.relu(self.dec3(x))
    x = F.relu(self.dec4(x))
    x = F.relu(self.dec5(x))
    x = F.relu(self.dec6(x))

    return x

class AE_3(nn.Module):
  def __init__(self, **kwargs):
    super(AE_3, self).__init__()

    # Encoder
    self.enc1 = nn.Linear(in_features=kwargs["input_shape"], out_features=512)  # Input image (28*28 = 784)
    self.enc2 = nn.Linear(in_features=512, out_features=256)
    self.enc3 = nn.Linear(in_features=256, out_features=128)
    self.enc4 = nn.Linear(in_features=128, out_features=64)
    self.enc5 = nn.Linear(in_features=64, out_features=32)
#    self.enc6 = nn.Linear(in_features=32, out_features=16)
    # Decoder
#    self.dec1 = nn.Linear(in_features=16, out_features=32)
    self.dec2 = nn.Linear(in_features=32, out_features=64)
    self.dec3 = nn.Linear(in_features=64, out_features=128)
    self.dec4 = nn.Linear(in_features=128, out_features=256)
    self.dec5 = nn.Linear(in_features=256, out_features=512)
    self.dec6 = nn.Linear(in_features=512, out_features=kwargs['input_shape'])  # Output image (28*28 = 784)

  def forward(self, x):
    x = F.relu(self.enc1(x))
    x = F.relu(self.enc2(x))
    x = F.relu(self.enc3(x))
    x = F.relu(self.enc4(x))
    x = F.relu(self.enc5(x))
#    x = F.relu(self.enc6(x))

#    x = F.relu(self.dec1(x))
    x = F.relu(self.dec2(x))
    x = F.relu(self.dec3(x))
    x = F.relu(self.dec4(x))
    x = F.relu(self.dec5(x))
    x = F.relu(self.dec6(x))

    return x

class AE_2(nn.Module):
  def __init__(self, **kwargs):
    super(AE_2, self).__init__()

    # Encoder
    self.enc1 = nn.Linear(in_features=kwargs["input_shape"], out_features=256)  # Input image (28*28 = 784)
    self.enc2 = nn.Linear(in_features=256, out_features=128)
    self.enc3 = nn.Linear(in_features=128, out_features=64)
    self.enc4 = nn.Linear(in_features=64, out_features=32)
    self.enc5 = nn.Linear(in_features=32, out_features=16)

    # Decoder
    self.dec1 = nn.Linear(in_features=16, out_features=32)
    self.dec2 = nn.Linear(in_features=32, out_features=64)
    self.dec3 = nn.Linear(in_features=64, out_features=128)
    self.dec4 = nn.Linear(in_features=128, out_features=256)
    self.dec5 = nn.Linear(in_features=256, out_features=kwargs['input_shape'])  # Output image (28*28 = 784)

  def forward(self, x):
    x = F.relu(self.enc1(x))
    x = F.relu(self.enc2(x))
    x = F.relu(self.enc3(x))
    x = F.relu(self.enc4(x))
    x = F.relu(self.enc5(x))

    x = F.relu(self.dec1(x))
    x = F.relu(self.dec2(x))
    x = F.relu(self.dec3(x))
    x = F.relu(self.dec4(x))
    x = F.relu(self.dec5(x))

    return x



class AE(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.encoder_hidden_layer = nn.Linear(
      in_features=kwargs["input_shape"], out_features=128
    )
    self.encoder_hidden_layer_2 = nn.Linear(
      in_features=128, out_features=128
    )
    self.encoder_output_layer = nn.Linear(
      in_features=128, out_features=128
    )
    self.decoder_hidden_layer = nn.Linear(
      in_features=128, out_features=128
    )
    self.decoder_hidden_layer_2 = nn.Linear(
      in_features=128, out_features=128
    )
    self.decoder_output_layer = nn.Linear(
      in_features=128, out_features=kwargs["input_shape"]
    )

  def forward(self, features):
    activation = self.encoder_hidden_layer(features)
    activation = torch.relu(activation)
    activation = self.encoder_hidden_layer_2(activation)
    activation = torch.relu(activation)
    code = self.encoder_output_layer(activation)
    code = torch.sigmoid(code)
    activation = self.decoder_hidden_layer(code)
    activation = torch.relu(activation)
    activation = self.decoder_hidden_layer(activation)
    activation = torch.relu(activation)
    activation = self.decoder_output_layer(activation)
    reconstructed = torch.sigmoid(activation)
    return reconstructed



