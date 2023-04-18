# MOT models
Here are files needed for the simulation of the MOT environment. They have been generated using data acquired on the experimental apparatus. 

## Fluorescence image generator
* MOT_fluo_img_generator.h5 : trained tensorflow CNN, input is a vector of number of atoms  and detuning. Number of atoms is normalized to (0,1). Detuning range is (0,1) with 1 corresponding to 50MHz (see look up tables below).

## Look up tables 

### Loading rate dN
* LUT_L.npy : measured loading rate in units of number of atoms
* det_L.npy : corresponding detuning values
* N_max.npy : maximal number of atoms encountered during training of the image generator, needed for normalization

### Temperature T
* LUT_T.npy : measured temperature in units of K, as a function of detuning
* det_T.npy : corresponding detuning values
 
