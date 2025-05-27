## About

Reconstructing the profile of a rough surface has many real world applications, such as for river monitoring and oceanography. The use of scattered acoustic waves for reconstruction is a promising method that has been investigated extensively, with many analytical and statistical approaches being used. This report details the use of computer aided algorithms motivated by Bayesian statistics to reconstruct arbitrary surface profiles, comparing a Hamiltonian Monte- Carlo (HMC) approach to a Bayesian Neural Network (BNN) approach. The models were tested using synthetic phaseless acoustic scattering data, produced through the Kirchhoff approximation for a variety of 2D, simple harmonic surfaces (consisting of a single sinusoid). Experimental validation was also carried out, taking data from an experimental setup with a known, machined surface. The BNN model performed well on synthetic surfaces at lower noise levels where only the amplitude and wavelength were recovered, but poorly when recovering surface offset and on experimental results. The HMC model performed well on synthetic surfaces at lower amplitudes (below 0.475cm) and was effective at recovering surface offset, however faced some inconsistencies in recovery. It also performed well on the experimental data with a surface error between 1 and 1.5 standard deviations of the true surface. This report serves to demonstrate the feasibility of such stochastic surface recovery approaches.

This project demonstrates the use of acoustic scattering to reconstruct rough surface profiles through the Kirchhoff approximation and stochastic techniques. The repository provides 2 reconstruction methods, a HMC method using the NUTS algorithm, and a BNN method. Example code for both is provided in the examples folder. Both methods have been ported to use the Jax library to utilize the GPU, which currently means the library will only work with GPU on Linux. It currently has been tested using an RTX 3070 on Linux Mint 21. This repository was used for my masters dissertation, with the state at the time of writing it contained in the final-report branch.

## Example Results

#### Single Harmonic HMC Reconstruction

![Reconstructed Surface](results/examples/3-parameter-hmc/nuts%20reconstruction.png)

![Parameters Corner Plot](results/examples/3-parameter-hmc/NUTS%20corner.png)

#### Multiple Harmonic HMC Reconstruction

![Reconstructed Surface2](results/examples/40-parameter-hmc/NUTS%20reconstruction.png)

## Build

#### Linux

1. Ensure the NVIDIA CUDA 12 library is installed.

2. Clone the repository 

3. Install the requirements listed in the requirements.txt file. These can be installed by running `python pip install -r requirements.txt`. It is    recommended to set up a virtual environment in the repository to install the packages to. 

4. Ensure the "libatlas-base-dev" library is installed so that pytensor can link for BLAS operations. On Debian systems for example this can be installed by running: ```sudo apt-get install libatlas-base-dev```

## Acknowledgements

I would like to extend my thanks to my project supervisor, Dr Anton Krynkin for providing invaluable feedback and resources throughout the year. I would also like to thank Michael David Johnson for providing the linearized Kirchhoff approximation code.