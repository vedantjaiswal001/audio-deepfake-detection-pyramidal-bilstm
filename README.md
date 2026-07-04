# Audio Deepfake Detection using Pyramidal BiLSTM

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success)

A PyTorch implementation of an enhanced **Pyramidal BiLSTM with Attention** for audio deepfake detection. The model utilizes **Log-Mel Spectrograms**, **hierarchical temporal downsampling**, and **multi-level data augmentation** for robust classification of real and AI-generated speech.

---

# Overview

Audio deepfake technology has become increasingly sophisticated, making reliable detection a critical research challenge.

This project implements a **Pyramidal BiLSTM architecture** that captures long-term temporal dependencies while progressively reducing sequence length. The system combines attention mechanisms with extensive data augmentation to improve detection performance.

---

# Features

- Pyramidal BiLSTM Architecture
- Attention Pooling
- Log-Mel Spectrogram Feature Extraction
- Audio Data Augmentation
- SpecAugment
- AdamW Optimizer
- OneCycle Learning Rate Scheduler
- Early Stopping
- Model Checkpointing
- Automatic Logging
- GPU (CUDA) Support

---

# Model Architecture

```
Audio
   │
   ▼
Log-Mel Spectrogram
   │
   ▼
BiLSTM
   │
   ▼
Pyramidal Downsampling
   │
   ▼
BiLSTM
   │
   ▼
Pyramidal Downsampling
   │
   ▼
BiLSTM
   │
   ▼
Attention Pooling
   │
   ▼
Fully Connected Layer
   │
   ▼
Sigmoid
   │
   ▼
Real / Fake
```

---

# Project Structure

```
audio-deepfake-detection-pyramidal-bilstm/
│
├── checkpoints/
├── logs/
├── results/
├── Audio_Deepfake_Detection.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Dataset

The model is trained on the **Fake-or-Real (FoR)** audio dataset.

Directory structure:

```
training/
validation/
testing/
```

---

# Feature Extraction

- Sampling Rate: 16 kHz
- Mono Audio
- Log-Mel Spectrogram
- 80 Mel Filters
- FFT Size: 1024
- Hop Length: 256

---

# Data Augmentation

### Audio-Level Augmentation

- Speed Perturbation
- Pitch Shifting
- White Noise Injection
- Pink Noise Injection

### Spectrogram-Level Augmentation

- Time Masking
- Frequency Masking (SpecAugment)

---

# Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Batch Size | 8 |
| Feature Type | Log-Mel Spectrogram |
| Mel Filters | 80 |
| Loss Function | Binary Cross Entropy |
| Early Stopping | Enabled |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/audio-deepfake-detection-pyramidal-bilstm.git
```

Navigate to the project:

```bash
cd audio-deepfake-detection-pyramidal-bilstm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python Audio_Deepfake_Detection.py
```

---

# Results

The proposed model provides:

- High classification accuracy
- Robust temporal feature learning
- Improved sequence representation through pyramidal downsampling
- Better generalization using multi-level augmentation
- Stable optimization with OneCycle Learning Rate scheduling

---

# Technologies

- Python
- PyTorch
- Librosa
- NumPy
- Scikit-learn
- Matplotlib
- CUDA

---

# Future Work

- Transformer-based architectures
- Self-supervised audio representation learning
- Real-time inference
- ONNX and TensorRT deployment
- Multi-language deepfake detection

---

# Author

**Vedant Jaiswal**

B.Tech, Computer Science and Engineering

Interests:
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Generative AI

---

# License

This project is licensed under the MIT License.
