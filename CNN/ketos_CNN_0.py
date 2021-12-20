"""
  Tryout some of the sample code -- here for ketos' CNN

  Note:  sample code has some 'typos'
  Original code:
          Examples:

            >>> # Most users will create a model based on a Ketos recipe
            >>> # The one below, specifies a CNN with 3 convolutional layers and 2 dense layers
            >>>
            >>> recipe = {'conv_set':[[64, False], [128, True], [256, True]], # doctest: +SKIP
            ...   'dense_set': [512, ],
            ...   'n_classes':2,
            ...   'optimizer': {'name':'Adam', 'parameters': {'learning_rate':0.005}},
            ...   'loss_function': {'name':'FScoreLoss', 'parameters':{}},
            ...   'metrics': [{'name':'CategoricalAccuracy', 'parameters':{}}]
            ... }
            >>> # To create the CNN, simply  use the  'build_from_recipe' method:
            >>> cnn = CNNInterface.build_from_recipe(recipe, recipe_compat=False) # doctest: +SKIP
  To get this to run, I renamed name to recipe_name  and put an underscore before build_from_recipe

"""

from ketos.neural_networks.cnn import CNNInterface
###########################################
def NNsummary(model_name, input_shape, recipe_name, nn):
    print("Model name is ", model_name, "\nRecipe name is ", recipe_name)
    print("***********************************************************************************************************")
    print("Layer Name     \t| Output Size    \t|  Kernal  \t| Strides \t|  Weights  \t| Biases  \t|Activation |Dropout|")
    print("Input array    \t|  {}  \t|         \t|        \t|             \t|        \t|       \t|      \t|".format(input_shape))
    cnt = 1
    pad = 0
    tot_weights = 0
    tot_biases = 0
    output_shape = [0,0,0]
    for layer in nn.convolutional_layers:
        n_filters = layer['n_filters']
        filter_shape = layer['filter_shape']
        padding = layer['padding']  # 'valid' means no padding and apply filters without going over edges
        if padding == 'valid':
            pad = 0
        strides = layer['strides']
        max_pool = layer['max_pool']
        activation = layer['activation']
        n_weights = int(filter_shape[0]*filter_shape[1]*input_shape[2] * n_filters)
        n_weights_str = "{:,}".format(n_weights).ljust(10)
        n_biases = n_filters
        tot_weights += n_weights
        tot_biases  += n_biases
        for i in range(2):
            output_shape[i] = int((input_shape[i] - filter_shape[i] +2*pad)/strides +1)
        output_shape[2] = n_filters
        shape = "({}, {}, {})".format(output_shape[0], output_shape[1], output_shape[2]).ljust(10)
        print("Conv-{}       \t|  {}  \t| {} \t|    {}  \t|  {}\t|    {}  \t|    {}  \t|      \t|".\
              format(cnt, shape, filter_shape, strides, n_weights_str, n_biases, activation) )
        if max_pool != None:
            pool_size = max_pool['pool_size']
            strides   = max_pool['strides']
            output_shape[0] = output_shape[0] // strides[0]
            output_shape[1]  = output_shape[1] // strides[1]
            shape = "({}, {}, {})".format(output_shape[0], output_shape[1], n_filters).ljust(10)
            print("MaxPool-{}    \t|  {}  \t| {}  \t| {}  \t|      0      \t|    0    \t|         \t|    \t|".format(cnt, shape, pool_size,strides))
        input_shape = output_shape
        cnt += 1
    for layer in nn.dense_layers:
        n_hidden = layer['n_hidden']
        dropout = layer['dropout']
        activation = layer['activation']
        n_weights = input_shape[0]*input_shape[1]*input_shape[2] * n_hidden
        n_biases = n_hidden
        tot_weights += n_weights
        tot_biases  += n_biases
        shape = "({}, {}, {})".format(output_shape[0], output_shape[1], n_hidden).ljust(13)
        n_weights_str = "{:,}".format(n_weights).ljust(10)
        print("Dense-{}       \t|  {}  \t|        \t|         \t|  {} \t|    {}  \t|    {} \t| {} \t|".\
              format(cnt, shape, n_weights_str, n_biases, activation, dropout))
    print("************************************************************************************************************")
    print("                                                   Totals \t|   {:,} \t|   {}  \t|".format(tot_weights, tot_biases))



############################################
# The recipe below, specifies a CNN with 3 convolutional layers and 2 dense layers
recipe1 = {'conv_set':[[64, False], [128, True], [256, True]],  'dense_set': [512, ], 'n_classes':2,\
          'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},\
          'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  \
          'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]}


recipe2 = {'convolutional_layers':\
    [{'n_filters':64, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', \
      'max_pool': None, 'batch_normalization':True}, \
     {'n_filters':128, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', \
      'max_pool': {'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True}, \
    {'n_filters':256, "filter_shape":(3,3), 'strides':1, 'padding':'valid', 'activation':'relu', \
          'max_pool': {'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True}\
     ],   'dense_layers': [{'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5}], \
           'n_classes':2, 'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},\
          'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  \
          'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
    }
recipe3 = {'convolutional_layers':\
    [{'n_filters':96, "filter_shape":(11,11), 'strides':4, 'padding':'valid', 'activation':'relu', \
      'max_pool': {'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True}, \
     {'n_filters':256, "filter_shape":(3,3), 'strides':2, 'padding':'valid', 'activation':'relu', \
      'max_pool': {'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True}, \
    {'n_filters':256, "filter_shape":(5,5), 'strides':2, 'padding':'valid', 'activation':'relu', \
          'max_pool': {'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True}\
     ],   'dense_layers': [{'n_hidden':4096, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5}], \
           'n_classes':2, 'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},\
          'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  \
          'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
    }

recipe4 = {'convolutional_layers':\
    [{'n_filters':8, "filter_shape":(5,5), 'strides':1, 'padding':'valid', 'activation':'relu', \
      'max_pool': {'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True}, \
     {'n_filters':16, "filter_shape":(5,5), 'strides':1, 'padding':'valid', 'activation':'relu', \
      'max_pool': {'pool_size':(2,2) , 'strides':(2,2)}, 'batch_normalization':True}, \

     ],   'dense_layers': [{'n_hidden':120, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5},\
                           {'n_hidden':84, 'activation':'relu', 'batch_normalization':True, 'dropout':0.5}], \
           'n_classes':2, 'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},\
          'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  \
          'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]
    }

cnn = CNNInterface._build_from_recipe(recipe3, recipe_compat=False)

model_name = "ketos_cnn_1"
recipe_name = 'recipe3'
input_shape = (256, 256, 3)
print(NNsummary(model_name, input_shape, recipe_name, cnn))

# extracted_recipe = cnn._extract_recipe_dict()
# print(extracted_recipe.keys())
# for key in extracted_recipe.keys():
#     print(key, ":", extracted_recipe[key])
# print("---------------")
# for layer in extracted_recipe['convolutional_layers']:
#     print(layer)


"""
Does ketos somewhere have a 'pretty print' for its models similar to keras.summary() which prints 
each layer in order something like:

Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 8, 8, 32)]       0                                                                        
 conv2d_transpose            (None, 24, 24, 32)       9216                                                                                                                              
 max_pooling2d               (None, 12, 12, 32)       0                                                                                                                                         
 re_lu (ReLU)                (None, 12, 12, 32)       0                                                                         
 conv2d_transpose_1          (None, 36, 36, 64)       18432                                                                                                                            
 max_pooling2d_1             (None, 18, 18, 64)       0                                                                 
.....

"""