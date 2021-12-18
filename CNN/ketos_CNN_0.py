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

# The recipe below, specifies a CNN with 3 convolutional layers and 2 dense layers
recipe = {'conv_set':[[64, False], [128, True], [256, True]],  'dense_set': [512, ], 'n_classes':2,\
          'optimizer': {'recipe_name':'Adam', 'parameters': {'learning_rate':0.005}},\
          'loss_function': {'recipe_name':'FScoreLoss', 'parameters':{}},  \
          'metrics': [{'recipe_name':'CategoricalAccuracy', 'parameters':{}}]}
# To create the CNN, simply  use the  'build_from_recipe' method:
cnn = CNNInterface._build_from_recipe(recipe, recipe_compat=False)

extracted_recipe = cnn._extract_recipe_dict()

print(extracted_recipe.keys())
for key in extracted_recipe.keys():
    print(key, ":", extracted_recipe[key])
print("---------------")
for layer in extracted_recipe['convolutional_layers']:
    print(layer)


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