# Noise-Resilient Heat Load Forecasting

This repository contains the implementation of my Master's thesis project:  
**"Predicting Heat Load in District Heating Systems – A Machine Learning Approach to Handling Noisy Data"**.  
The work explores hybrid deep learning architectures that integrate denoising autoencoders with LSTM networks to improve short-term heat load forecasting under noisy IoT data conditions.

---

## Overview
District Heating (DH) systems play a critical role in sustainable urban energy.  
However, IoT-collected data is often noisy due to sensor faults, communication errors, and missing values.  
This project addresses this challenge by combining **denoising mechanisms** with **time-series forecasting models** to improve robustness and accuracy.

---

## Key Steps and Components

### 1. Data Collection and Preprocessing
- **Source:** IoT-enabled sensors from a DH-powered community in Großschönau, Austria.  
- **Parameters:** supply water temperature, return water temperature, and water flow.  
- **Preprocessing:**
  - Aggregated raw 1-minute data into 15-minute intervals.
  - Calculated delivered heat using thermodynamic equations.
  - Handled missing values (shift + forward fill).
  - Removed outliers using the Inter-Quartile Range (IQR).
  - Engineered time-based features (hour/day cyclic transforms, holidays/weekends).
  - Normalized features with Min-Max scaling.
  - Generated sliding-window sequences for supervised learning.

### 2. Predictive Models
Three architectures were implemented and compared:
- **Baseline LSTM:** sequence-to-sequence forecasting model.  
- **DAE-LSTM:** integrates a denoising autoencoder with LSTM forecasting.  
- **CDAE-LSTM:** combines convolutional layers (for local feature extraction) with denoising and LSTM forecasting.  

### 3. Training & Optimization
- Implemented with **TensorFlow/Keras**.  
- Optimized via **Keras Tuner RandomSearch**.  
- Early stopping applied to prevent overfitting.  
- Adam optimizer used for training.

### 4. Model Evaluation
- Test data perturbed with **5%, 10%, and 15% Gaussian noise**.  
- Evaluation metrics: **Mean Absolute Error (MAE)** and **Mean Absolute Percentage Error (MAPE)**.  
- Results:
  - Denoising models consistently outperformed the baseline.  
  - Example (Hotel dataset, 15% noise):
    - Baseline LSTM → 21.2% MAPE  
    - DAE-LSTM → 8.6% MAPE  
    - CDAE-LSTM → 7.9% MAPE  

---

## Tools and Environment
- **Python:** NumPy, Pandas, Scikit-learn, Matplotlib, Statsmodels  
- **Deep Learning:** TensorFlow/Keras, Keras Tuner  
- **Development:** Jupyter Notebook on macOS Sonoma (M2 chip)  
