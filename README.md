# BayesianDI

## Overview
This repository provides a Python implementation for estimating stellar parameters and surface maps from time-series spectra. It utilizes **JAX** for high-performance computation and **NumPyro** for MCMC sampling, specifically designed for Doppler Imaging analysis (e.g., Luhman 16B).

## Prerequisites
- **Python 3.9.13**

## Installation

1. **Install basic dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **GPU Support (Recommended)**
   For faster MCMC sampling, it is highly recommended to install the GPU-enabled version of JAX separately, depending on your CUDA environment.
   For CUDA 12,
   ```
   pip install -U "jax[cuda12]"
   ```

## Usage

1. **Parameter Estimation**
   Run `estimate.py` to perform MCMC sampling for stellar parameters.
2. **Surface Map Reconstruction**
   Run `moments.py` to generate posterior mean and variance maps.
