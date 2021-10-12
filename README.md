# orca-autoencoder

## directory structure for this autoencoder project

My toplevel directory on my **local machine** is titled 'github'


```
                              github
                                |
            ---------------------------------------------
            |                   |             |          |
   orca-autoencoder-codes   spectrograms    models     jpgs
            |
    -----------------
    |       |       |
   MLP     CNN   analyzers	
    |
orca-autoencoder_MLP.py
```
  
  The python scripts are maintained under github in the `orca-autoencoder-codes` sub-directory.
  
  * MLP = Multilevel Perceptron scripts
  * CNN = Convolutional Neural Network scripts

  The rest of this project is stored outside of Github:
  
  * input data for the neural networks are stored in the `spectrograms` sub-directory
  * trained models are stored in the `models` sub-directory;
  * various scripts create jpg files and these are stored in the `jpgs` sub-directry.
  
  **orca-autoencoder_MLP.py** is a fully connected multilayer perceptron network
  
   
  
    
