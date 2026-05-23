# Heart Rate Estimation from NIR Facial Videos
## Remote Photoplethysmography (rPPG) using PhysNetGRU
 
**Dataset:** MR-NIRP-D (MERL-Rice NIR Pulse Dataset — Driving)  
**Model:** PhysNetGRU (3D-CNN + Bidirectional GRU)  
**Framework:** PyTorch  

---

## Project Overview

This project implements a deep learning pipeline to estimate **Heart Rate (BPM)** from Near-Infrared (NIR) facial video sequences using Remote Photoplethysmography (rPPG).

The system:
1. Streams NIR frames directly from zip archives (no disk extraction)
2. Extracts ground truth BPM from pulse oximeter `.mat` files via bandpass filtering + FFT
3. Trains a PhysNetGRU model (3D-CNN + Bidirectional GRU) to predict rPPG waveforms
4. Derives final BPM via FFT on the predicted waveform

**Final Results:**
| Metric | Value |
|--------|-------|
| MAE | 10.94 BPM |
| RMSE | 15.53 BPM |
| Pearson r | 0.203 (p=0.006) |
| Test samples | 180 clips |

---

## 📁 Repository Structure

```
├── NIR_HR_Estimation.ipynb   ← Main Colab notebook (all code + outputs)
├── README.md                 ← This file
├── Report.docx               ← Written report with architecture diagram
└── outputs/
    ├── training_curves.png   ← Loss/MAE/RMSE per epoch
    └── evaluation_results.png← Scatter plot + error distribution
```

---

## 🗂️ Dataset Structure (Discovered)

```
MR-NIRP Car/                          ← Google Drive shared folder
  Subject1/
    subject1_driving_still_940/
      NIR.zip     → NIR/Frame00001.pgm ... Frame12600.pgm  (16-bit PGM)
      PulseOX.zip → PulseOX/pulseOx.mat                   (ground truth)
      RGB.zip     → NOT used (NIR-only per assignment rules)
    subject1_driving_small_motion_940/
    ...
  Subject2/
  ...
  Subject19/
```

**Key facts:**
- 19 subjects × ~10 clips each = 185 total clip folders
- NIR frames: 16-bit PGM grayscale images (~10,000–12,600 frames per clip)
- Ground truth: pulse oximeter BVP signal in `.mat` format inside `PulseOX.zip`
- Conditions: driving (still / small motion / large motion) × two wavelengths (940nm / 975nm)

---

## Setup & Dependencies

### Requirements
```
Python 3.10+
torch >= 2.0
torchvision
scipy
opencv-python-headless
matplotlib
scikit-learn
tqdm
numpy
```

### Install
```bash
pip install torch torchvision scipy opencv-python-headless matplotlib scikit-learn tqdm numpy
```

---

## 🚀 How to Run (Google Colab)

1. **Add dataset shortcut to your Drive:**
   - Open: https://drive.google.com/drive/folders/1U3fzIOESmaBAyikGF0cKI2wW3YK8JqCK
   - Right-click folder → "Add shortcut to Drive" → My Drive → Add

2. **Open notebook in Colab:**
   - Upload `NIR_HR_Estimation.ipynb` to Colab
   - Runtime → Change runtime type → **T4 GPU**

3. **Run all cells top to bottom:**
   - Section 0: Install dependencies (~1 min)
   - Section 1: Mount Drive + set dataset path (~30 sec)
   - Section 2: Imports
   - Section 3: Discover clip folders
   - Section 4: Ground truth preprocessing + test
   - Section 5: NIR frame loader + visualization
   - Section 6: Build datasets (~20 min — reads from Drive)
   - Section 7: Define PhysNetGRU model
   - Section 8: Define loss functions
   - Section 9: Train 30 epochs (~2 hours on T4 GPU)
   - Section 10: Plot training curves
   - Section 11: Evaluate on test set
   - Section 12: Save all outputs

---

## Model Architecture

```
Input: [B, 1, T=150, H=64, W=64]
│
├─ ConvBlock3D(1→16,  kernel=1×5×5)   spatial feature extraction
├─ ConvBlock3D(16→32, kernel=3×3×3)   spatiotemporal features
├─ MaxPool3d(1,2,2)                    spatial: 64→32
│
├─ ConvBlock3D(32→64, kernel=3×3×3)
├─ ConvBlock3D(64→64, kernel=3×3×3)
├─ MaxPool3d(1,2,2)                    spatial: 32→16
│
├─ ConvBlock3D(64→64, kernel=3×3×3)
├─ MaxPool3d(1,4,4)                    spatial: 16→4
├─ Dropout3d(0.3)
│
├─ Reshape: [B, T, 1024]
├─ Bidirectional GRU(1024→128)        temporal cardiac rhythm
├─ Linear(128→1)
│
Output: [B, T] rPPG waveform
        → FFT → dominant cardiac frequency → BPM
```

**Total parameters:** 784,369

---

## Ground Truth Preprocessing Pipeline

```
PulseOX.zip
  └── PulseOX/pulseOx.mat
        │
        ▼
   Load raw BVP signal array
        │
        ▼
   Estimate pulse ox sampling rate:
   gt_fs = signal_length / video_duration
        │
        ▼
   Resample to video frame count
   (scipy.signal.resample)
        │
        ▼
   Normalize: zero-mean, unit variance
        │
        ▼
   Bandpass filter: 0.7–4.0 Hz
   (= 42–240 BPM cardiac range)
   Removes: respiration (<0.3 Hz)
            motion noise (>4 Hz)
        │
        ▼
   FFT → find peak frequency
   in cardiac band (0.7–4.0 Hz)
        │
        ▼
   BPM = peak_frequency × 60
```

---

## Loss Function

**Combined Loss = 0.7 × Pearson Loss + 0.3 × Huber Loss**

- **Pearson Loss** `(1 - r)`: Trains waveform shape correlation — standard in rPPG literature (PhysNet, DeepPhys)
- **Huber Loss** on BPM: Directly minimizes BPM error, robust to outliers

---

## Evaluation Metrics

| Metric | Description | Result |
|--------|-------------|--------|
| MAE | Mean Absolute Error (BPM) | 10.94 |
| RMSE | Root Mean Square Error (BPM) | 15.53 |
| Pearson r | Correlation between predicted and GT HR | 0.203 (p=0.006) |

---

## Errors Encountered & Fixed

| Error | Root Cause | Fix |
|-------|-----------|-----|
| 0 NIR videos found | Code looked for `.avi` files; dataset has `.pgm` in zips | Rewrote loader for zip streaming |
| `cv2.imdecode()` returning None | 16-bit PGM format not supported by `IMREAD_GRAYSCALE` | Used `IMREAD_ANYDEPTH` + normalize `/256` |
| MAE stuck at 22.37 BPM | FFT resolution limited by clip length | Documented as limitation |
| Runtime disconnection during loading | 20+ min Drive streaming lost on disconnect | Pickle save to Drive after loading |

---

## References

- Chen, W., & McDuff, D. (2018). DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks. ECCV.
- Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics Express.
- Hu, M., et al. (2021). MR-NIRP: A Multi-Resolution NIR rPPG Dataset. IEEE TBIOM.
