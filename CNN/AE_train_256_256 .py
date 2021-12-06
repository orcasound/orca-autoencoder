import helpers_3D as d3
import time

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import ReLU, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras

import ketos.data_handling.database_interface as dbi
from ketos.data_handling.data_feeding import BatchGenerator

#######################################

def encoderAE(inputs, layers):
    """ Construct the Encoder
        inputs : the input vector
        layers : number of filters per layer    """
    x = inputs
    # Feature pooling by 1/2H x 1/2W
    for n_filters in layers:
        x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    return x

def decoderAE(x, layers):
    """ Construct the Decoder
      x      : input to decoder
      layers : the number of filters per layer (in encoder)
    """
    # Feature unpooling by 2H x 2Wdec
    for _ in range(len(layers) - 1, 0, -1):
        n_filters = layers[_]
        x = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False,
                            kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    # Last unpooling, restore number of channels
    x = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

########################################
def countCases(dataset, data, annot):
  tot = len(data[:]['label'])
  ones = sum(data[:]['label'])
  zeros = tot - ones
  print(dataset, ": N calls =", ones, "N backs=", zeros, "Total =", tot, "Annots", len(annot[:]))

#####################################################  RUN RUN RUN



#######################  User set parameters VVVVVVVVVVVVVVV below

loadAE_ModelFilename = ""

h5filename = 'Tinydb_h5_256_256__09_12_2021_wavs.h5'

#saveAE_ModelDir = baseDir + "models_256/{}_epochs_{}/".format(newModelID, addnl_epochs)
#saveEncoder_ModelDir = baseDir + "models_256/Encoder_{}/".format(newModelID)

addnl_epochs = 10
#######################  User set parameters ^^^^^^^^^^^^^^^^^^^  above


h5file = dbi.open_file(h5filename, 'r')
train_data = dbi.open_table(h5file, "/train/table_data")
train_annot = dbi.open_table(h5file, "/train/table_annot")
val_data = dbi.open_table(h5file, "/eval/table_data")
val_annot = dbi.open_table(h5file, "/eval/table_annot")
test_data = dbi.open_table(h5file, "/test/table_data")
test_annot = dbi.open_table(h5file, "/test/table_annot")

############## THIS LOADS ENTIRE h5 DATASET ####   SLOW!!!!
countCases('train', train_data, train_annot)
countCases('val', val_data, val_annot)
countCases('test', test_data, test_annot)

#Create a BatchGenerator from a data_table and separate annotations in a anot_table
train_generator = BatchGenerator(data_table=train_data, annot_in_data_table=False, annot_table=train_annot,\
                                 batch_size=3, x_field='data', return_batch_ids=True) #create a batch generator
val_generator = BatchGenerator(data_table=val_data, annot_in_data_table=False, annot_table=val_annot,\
                                 batch_size=3, x_field='data', return_batch_ids=True)

# Setup Encoder taking input spectrograms and outputting feature sets at the bottleneck layer
# Setup Decoder taking bottlenect features and recreating the input spectrogram
# metaparameter: number of filters per layer in encoder
layers = [256, 128, 64, 32, 32]
# The input tensor
inputs = Input(shape=(256, 256, 1))
# The encoder
bottleneck = encoderAE(inputs, layers)
encoder_Model = Model(inputs, bottleneck)
# The decoder
outputs = decoderAE(bottleneck, layers)

# Instantiate the Model
if loadAE_ModelFilename != "":
    aeAE_Model = keras.models.load_model(loadAE_ModelFilename)
    aeAE_Model.compile(loss='binary_crossentropy', optimizer='adam',
                       metrics=['accuracy'])  # adam default learning_rate is 0.001
    score = aeAE_Model.evaluate(test_data, test_data)
    print('-----------------------Starting with Test loss:', score[0])
    print('-----------------------Starting with Test accuracy:', score[1])
else:
    aeAE_Model = Model(inputs, outputs)
    aeAE_Model.compile(loss='binary_crossentropy', optimizer='adam',
                       metrics=['accuracy'])  # adam default learning_rate is 0.001

print(aeAE_Model.summary())
tstart = time.time()
###  Run the model  ---- For Ketos, I want to do something like:

    # >>> recipe = {'conv_set':[[64, False], [128, True], [256, True]],
    # ...   'dense_set': [512, ],
    # ...   'n_classes':2,
    # ...   'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
    # ...   'loss_function': {'name':'FScoreLoss', 'parameters':{}},
    # ...   'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]
    # ... }
    # >>> # To create the CNN, simply  use the  'build_from_recipe' method:
    # >>> cnn = CNNInterface.build_from_recipe(recipe, recipe_compat=False)
    # then include train_generator, val_generator, checkpoint_dir
    # and eventually execute a train_loop(n_epochs = A_Nice_Number)

###  Done running model
# print("save aeAE_Model {} at directory {}".format(newModelID, saveAE_ModelDir))
# aeAE_Model.save(saveAE_ModelDir)
# encoder_Model.save(saveEncoder_ModelDir)


tstop = time.time()
print("Elapsed time s {}, m {}, hr {:.2f} s/epoch {:.2f} ".format(int(tstop - tstart), int((tstop - tstart) / 60.0),
                                                                  ((tstop - tstart) / 3600.0),
                                                                  (tstop - tstart) / addnl_epochs))
##########################

