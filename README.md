# **Scattering correction for infrared spectra of biological cells using computational infrared microspectroscopy and deep learning**

*Sergio G. Rodrigo $^{1,2*}$, Ilia L. Rasskazov $^{3}$, Luis Martin-Moreno $^{1,4}$, Martin Schnell5 $^{6,7*}$*



+ 1 Instituto de Nanociencia y Materiales de Aragón (INMA), CSIC-Universidad de Zaragoza, Zaragoza, 50009, Spain
+ 2 Departamento de Física Aplicada, Universidad de Zaragoza, Zaragoza, 50009, Spain
+ 3 Independent Researcher, San Jose, California 95124, United States
+ 4 Departamento de Física de la Materia Condensada, Universidad de Zaragoza, Zaragoza, 50009, Spain
+ 5 CIC nanoGUNE BRTA, 20018 Donostia-San Sebastián, Spain
+ 6 Donostia International Physics Center, Donostia-San Sebastian 20018, Spain
+ 7 IKERBASQUE, Basque Foundation for Science, 48013 Bilbao, Spain

$^*$ Corresponding authors. Email: sergut@unizar.es (SGR), schnelloptics@gmail.com (MS)

## Typical workflow

### 1) Train the deep-learning surrogates for the FDTD calculations of the absorbance spectrum, $A(\nu)$ , and the true absorption spectrum, $\mathcal{A}_{true} (\nu)$.

```bash
fdtd_surrogate_neural_network_model.ipynb
```
 + Repeat this step twice: first for absorption and then for absorbance.
 + Two .keras files will be generated: surrogate_neural_network_model_absorbance.keras and surrogate_neural_network_model_absorption.keras

 This notebook implements and evaluates deep learning models for scattering correction in infrared spectra of biological cells. It uses artificial neural networks as surrogates for FDTD calculations to predict absorbance or absorption spectra.

Key points include:

Objective: To develop deep learning models to correct scattering in infrared spectra, specifically for biological cells, acting as surrogates for FDTD simulations.
Models: Two identical feed-forward neural network models, implemented with Keras and TensorFlow, are used.
Model Input: Each model takes a parameter vector $p=(r_x, r_y, r_z, k_m)$$p=(r_x, r_y, r_z, k_m)$ as input, representing the semi-axes of an ellipsoid and a scaling factor for the imaginary refractive index.
Model Output: A vector of 251 values, corresponding to the spectral wavelengths of the spectrum to be predicted (absorbance or true absorption).
Data Processing: Simulation data is loaded, randomized, normalized, and split into training (80%) and validation (20%) sets.
Neural Network Architecture: Consists of an input layer, four hidden dense layers with sigmoid activation and Batch Normalization, and a linear output layer.
Training: Models are trained for 1000 epochs with a batch size of 64, using the RMSprop optimizer and the mean squared error (MSE) loss function. The best model is saved during training.
Evaluation: The performance of the predicted neural network is compared with the true values from the validation set, both for the entire set and for randomly selected individual spectra.

### 2) Train the deep-learning model that predicts true absorption (heat dissipation) spectra, $\mathcal{A}_{true} (\nu)$, from absorbance spectra, $A(\nu)$.

```bash
encoder_decoder_neural_network_model.ipynb
```
+ One .keras file will be generated: encoder_decoder_network_model.keras

### 3) Inverse design optimization.

```bash
inverse_design_optimization.ipynb
```
+ The .keras files generated are loaded. 