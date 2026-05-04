# ECE_5424_Capstone_Project

Fairness-aware hand-load estimation from wearable biomechanical sensors. The project compares a 1D-CNN Variational Autoencoder (VAE) regressor against a Debiasing VAE (DVAE) with a gradient-reversal adversarial objective, on both forearm EMG and full-body IMU collected during free-style lifting. Evaluation covers accuracy (MAE, RMSE, nearest-class) and a multi-metric sex-fairness suite (statistical parity, positive/negative residual disparity, disparate impact, bounded group loss).

## Project phases

**Phase 1 — Random Forest baseline (EMG).** Handcrafted time- and frequency-domain features into RF regressor/classifier; LOSO evaluation; preliminary sex-bias analysis. Established that handcrafted features generalize poorly across subjects under unconstrained lifting, motivating the move to learned representations.

**Phase 2 — VAE / DVAE (EMG and IMU).** 1D convolutional encoder, mirrored ConvTranspose decoder, MLP regressor head. The DVAE adds a parallel sex-specific encoder and two adversarial heads through a gradient-reversal layer, so one latent stays sex-agnostic and feeds the weight regressor at inference. Subject-independent, sex-stratified 80/20 split.

## Repository layout

  
```
project-root/
├── README.md
├── requirements.txt
├── RF/ # Phase 1: RF baseline (EMG)
│ ├── Data Process-1.ipynb
│ ├── RF_Model_and_Analysis.ipynb
│ └── RF_Regressor_Classifier.ipynb
└── VAE/ # Phase 2: VAE / DVAE
  │
  │ # Preprocessing (run first)
  ├── 00_filter_emg.ipynb # EMG bandpass + notch filtering
  ├── 01_segment_lifts.ipynb # cycle segmentation, NaN cleanup
  │
  │ # VAE regressor (one per modality)
  ├── 02_vae_imu.ipynb # IMU VAE — pilot (reconstruction only)
  ├── 03_vae_emg.ipynb # EMG VAE — pilot (reconstruction only)
  ├── 04_vae_imu_reg.ipynb # IMU VAE regressor (final)
  ├── 05_vae_emg_reg.ipynb # EMG VAE regressor (final)
  │
  │ # DVAE — adversarial sex-debiasing
  ├── 06_dvae_imu.ipynb # IMU DVAE
  ├── 07_dvae_emg.ipynb # EMG DVAE
  │
  │ # Cross-model fairness analysis
  └── 08_fairness_analysis.ipynb # SP, PRD/NRD, DI, BGL across all four models
```
