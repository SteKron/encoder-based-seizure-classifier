# Seizure Classification Model

## Overview
This repository contains a model for seizure classification on the CHB-MIT dataset. The model is based on the encoder component of BrainCodec[1].

## Model Architecture
The model is composed of three main elements:

1. **Encoder**: The pretrained encoder from BrainCodec is used to extract temporal embeddings from raw EEG signals. The autoencoder was designed following principles outlined in prior research on self-supervised learning for biosignal compression and feature extraction.
2. **Spatial Fusion Block**: A 2D convolutional layer is applied to fuse information across EEG channels, inspired by the approach of SpatioSpectral representation learning [2]. This is followed by an MLP that captures hierarchical spatiotemporal relationships.
3. **Linear Classifier**: A fully connected layer classifies the processed embeddings into seizure/non-seizure labels.

## Labels Loading
Seizure labels are loaded at a frequency of **1Hz**. However, since the encoder processes EEG windows at **0.25Hz**, each window contains **four labels**. Each label is determined based on seizure event coverage:
- A 1-second segment is labeled as a seizure if a seizure event covers more than **50%** of its duration, as suggested by the SzCORE framework [3].

## Scripts
This repository includes two main scripts for model evaluation:

1. **Standard Model Testing**: Runs inference and evaluates model performance.
2. **Leave-One-Subject-Out Cross-Validation (LOOCV)**: This script loops around PyTorch Lightning to generate subject-independent validation results. LOOCV was tested using data from **at most two subjects** due to memory constraints.

## Memory Considerations
While running tests, it was observed that loading EEG data for multiple subjects exceeded memory limitations. This occurs because the **BIDS loader** loads all subject data into memory at once. A potential solution is implementing **lazy loading**, but initial tests indicated that this approach was excessively slow. Therefore, it was not included in this release.

## References
1. [The Case for Cleaner Biosignals: High-Fidelity Neural Compression for EEG/iEEG Transfer](https://openreview.net/forum?id=b57IG6N20B).
2. [SpatioSpectral Representation Learning for EEG gait-pattern classification](https://ieeexplore.ieee.org/document/8428659).
3. [SzCORE: A Seizure Community Open-source Research Evaluation framework](https://onlinelibrary.wiley.com/doi/10.1111/epi.18113).
