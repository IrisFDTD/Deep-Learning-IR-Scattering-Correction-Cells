# **Scattering correction for infrared spectra of biological cells using computational infrared microspectroscopy and deep learning**

*Sergio G. Rodrigo $^{1,2*}$, Ilia L. Rasskazov $^{3}$, Luis Martin-Moreno $^{1,4}$, Martin Schnell $^{5,6,7*}$*



+ 1 Instituto de Nanociencia y Materiales de Aragón (INMA), CSIC-Universidad de Zaragoza, Zaragoza, 50009, Spain
+ 2 Departamento de Física Aplicada, Universidad de Zaragoza, Zaragoza, 50009, Spain
+ 3 Independent Researcher, San Jose, California 95124, United States
+ 4 Departamento de Física de la Materia Condensada, Universidad de Zaragoza, Zaragoza, 50009, Spain
+ 5 CIC nanoGUNE BRTA, 20018 Donostia-San Sebastián, Spain
+ 6 Donostia International Physics Center, Donostia-San Sebastian 20018, Spain
+ 7 IKERBASQUE, Basque Foundation for Science, 48013 Bilbao, Spain

$^*$ Corresponding authors. Email: sergut@unizar.es (SGR), schnelloptics@gmail.com (MS)

## Workflow

### 1) Train the deep-learning surrogates for the FDTD calculations of the absorbance spectrum, $A(\nu)$ , and the true absorption spectrum, $\mathcal{A}_{true} (\nu)$.

```bash
fdtd_surrogate_neural_network_model.ipynb
```
The main goal of this notebook is to train artificial neural networks as surrogates for FDTD calculations to predict absorbance or absorption spectra.

The process involves:
+ Loading Data: Importing `fdtd_simulations.mat`, that contains 11,000 absorbance and absorption spectra computed with the finite-difference time-domain (FDTD) method for ellipsoids with different shapes and refractive indices.
+ Data Processing: Simulation data is loaded, randomized, normalized, and split into training (80%) and validation (20%) sets.
+ Model Definition: A feed-forward neural network model is implemented with Keras and TensorFlow. Consists of an input layer, four hidden dense layers with sigmoid activation and Batch Normalization, and a linear output layer. Each model takes a parameter vector $p=(r_x, r_y, r_z, k_m)$ as input, representing the semi-axes of an ellipsoid and a scaling factor for the imaginary refractive index. The output is a vector of 251 values, corresponding to the spectral wavelengths of the spectrum to be predicted (absorbance or true absorption). 
+ Training: Models are trained for 1000 epochs with a batch size of 64, using the RMSprop optimizer and the mean squared error (MSE) loss function. The best model is saved during training.
+ Evaluation: The performance of the predicted neural network is compared with the true values from the validation set, both for the entire set and for randomly selected individual spectra.
+ Run the notebook twice: first for absorption and then for absorbance. Two *.keras* files will be generated: surrogate_neural_network_model_absorbance.keras and surrogate_neural_network_model_absorption.keras


### 2) Train the deep-learning model that predicts true absorption (heat dissipation) spectra, $\mathcal{A}_{true} (\nu)$, from absorbance spectra, $A(\nu)$.

```bash
encoder_decoder_neural_network_model.ipynb
```

The main goal is to predict true absorption spectra from observed absorbance spectra using a neural network. 

The process involves:
+ Loading Data: Importing `fdtd_simulations.mat` that contains 11,000 absorbance and absorption spectra computed with the finite-difference time-domain (FDTD) method for ellipsoids with different shapes and refractive indices.
+ Data Processing: Shuffling and splitting the data into 90% for training and 10% for validation.
+ Model Definition: Building a custom encoder-decoder model with specified neuron counts for each layer, including a latent space. It utilizes a fully connected encoder-decoder network architecture with three encoding and three decoding layers, designed to handle 251 spectral wavelengths.
+ Training: Compiling and fitting the model using the Adam optimizer and Mean Squared Error (MSE) as the loss function, over 1000 epochs, with early stopping and model checkpointing to save the best performing model.
+ Evaluation: Loading the trained model and comparing its predictions on the validation set against the true absorption values, both for the entire set (scatter plot) and for individual, randomly selected spectra.
+ A *.keras* file will be generated: encoder_decoder_network_model.keras

### 3) Inverse design optimization.

```bash
inverse_design_optimization.ipynb
```

This notebook focuses on an inverse design process. It employs a gradient-descent algorithm to find optimal physical parameters (ellipsoid semi-axes and a material scaling factor) that minimize a defined objective function. This objective function compares predicted and target infrared spectra (absorbance and absorption).

The notebook utilizes pre-trained neural network models (the *.keras* files generated):
+   Surrogate models for FDTD simulations predict absorbance and true absorption from the design parameters.
+   An encoder-decoder model infers a physically plausible true absorption spectrum from target absorbance.

It loads:
+ File `fdtd_simulations.mat`, which contains 11,000 absorbance and absorption spectra computed with the FDTD method for ellipsoids with different shapes and refractive indices.
+ Several files with format `model_*fsp_spectra`, which contains the absorbance and absorption spectra of the three HeLa cells investigated in this work as computed with the FDTD method. The files are organized according to the simulated region (cell nucleus or entire cell) and the refractive index used in each simulation

The workflow includes loading HeLa cell spectral data, preparing training datasets for the neural networks, loading the pre-trained models, and then performing an inverse design optimization. This optimization involves multiple runs of a gradient descent algorithm to find parameter sets that best match the target HeLa cell spectra. Finally, the results are sorted by cost, and the predicted parameters and spectra are visualized, including 2D projection maps of the optimized physical parameters.
