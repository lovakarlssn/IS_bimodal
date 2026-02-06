# IS_bimodal: Bimodal (EEG + fMRI) Data Augmentation & Fusion

## Project Roadmap

This research is divided into four distinct phases. We are currently in **Phase 1**.

- [x] **Phase 1: Augmentation** (Current Focus)
    EEG Augmentation:
    - Benchmarking spatial, temporal, and frequency-based augmentations.
    fMRI Augmentations:
    - Benchmarking ... augmentations.

    
- [ ] **Phase 2: Classification**
    - Fusion Strategy Benchmarking
    - Measure fusion gain
    - Benchmark different fusion strategies
- [ ] **Phase 3: Generalization (Transfer Learning)**
    - Evaluating cross-dataset generalization
- [ ] **Phase 4: Final Evaluation**
    - All models will be evaluated using two schemes:
        - Intra-Subject: 5-fold Cross Validation
        - Inter-Subject: Leave-One-Subject-Out (LOSO)
---


## Repository Structure

```text
IS_bimodal/
├── augmentations/       # Augmentation logic (EEG & future fMRI)
│   └── eeg_aug.py       # Spatial Shuffle, Time Slice, Freq Surrogate
├── data/                # .npy files - **NOT UPLOADED**
├── experiments/         # Training engines (Cross-Validation, Full Training)
├── loaders/             # Data loading
├── models/              # Model architectures
├── notebooks/           # Analysis & Visualization
├── utils/               # Helper metrics and plotting
├── config.py            # Global Hyperparameters
└── main.py              