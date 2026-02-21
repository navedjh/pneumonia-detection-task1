# Pneumonia Detection from Chest X-rays using Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navedjh/pneumonia-detection-task1/blob/main/notebooks/pneumoniadetection_complete.ipynb)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![MedMNIST](https://img.shields.io/badge/MedMNIST-v2-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Project Overview

This repository contains a complete implementation of a pneumonia detection system using the **PneumoniaMNIST dataset** from MedMNIST v2. The project implements multiple deep learning architectures with comprehensive analysis, ensemble methods, and interpretability via **Grad-CAM visualization**.

### 🎯 Task Description
Build a convolutional neural network (CNN) classifier for pneumonia detection with thorough performance analysis, including:
- Complete data pipeline with preprocessing and augmentation
- Multiple model architectures (CNN, Vision Transformer, Ensemble)
- Comprehensive evaluation metrics
- Failure case analysis
- Model interpretability with Grad-CAM

### 📊 Dataset Characteristics
| Split | Total | Normal | Pneumonia | Ratio |
|-------|-------|--------|-----------|-------|
| **Training** | 4,708 | 1,214 | 3,494 | 2.88:1 |
| **Validation** | 524 | 135 | 389 | 2.88:1 |
| **Test** | 624 | 234 | 390 | 1.67:1 |

- **Image format**: 28×28 grayscale images
- **Source**: Chest X-ray images (MedMNIST v2)
- **Task**: Binary classification (Normal vs. Pneumonia)

## 🏆 Key Results (30 Epochs)

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Missed Cases |
|-------|----------|-----------|--------|----------|---------|--------------|
| **CNN** | **88.14%** | 85.75% | **97.18%** | **91.11%** | **96.49%** | **11** |
| **ViT** | 78.85% | 76.88% | 94.62% | 84.83% | 90.50% | 21 |
| **Ensemble** | 86.54% | 83.85% | **97.18%** | 90.02% | 95.53% | **11** |

### 🏥 Clinical Impact
- **CNN and Ensemble miss only 11 out of 390 pneumonia cases** (2.82% miss rate)
- **97.18% detection rate** - clinically excellent for screening
- **False positives**: 82-95 cases (acceptable for triggering follow-up tests)
### 🏥 Clinical Impact
- **CNN and Ensemble miss only 11 out of 390 pneumonia cases** (2.82% miss rate)
- **97.18% detection rate** - clinically excellent for screening
- **False positives**: 82-95 cases (acceptable for triggering follow-up tests)

### Confusion Matrices

#### CNN Confusion Matrix
<img width="737" height="587" alt="cnnconfusionmatrix" src="https://github.com/user-attachments/assets/5ed4104f-d8af-4be4-b4cc-08ff3153cf96" />

          Predicted
          Normal  Pneumonia
        
<img width="1436" height="591" alt="CNNfalsepositive" src="https://github.com/user-attachments/assets/af9cb31b-dd2b-4f4d-8d14-23a56e77b620" />

#### Ensemble Confusion Matrix

<img width="737" height="587" alt="Ensembleconfusionmatrix" src="https://github.com/user-attachments/assets/9df10e06-494f-4867-95da-42fa1100a2fb" />



## 🛠️ Models Implemented

### 1. **Custom CNN** (Best Individual Model)
- 3 convolutional blocks with batch normalization
- Progressive feature extraction
- Dropout regularization (0.25-0.3)
- Global average pooling
- **Parameters**: ~150K

### 2. **Vision Transformer (ViT)**
- Patch embedding (7×7 patches, stride 4)
- Position embeddings
- Multi-head self-attention (4 heads)
- MLP head
- **Parameters**: ~170K

### 3. **Ensemble**
- Average of CNN and ViT predictions
- Combines local (CNN) and global (ViT) features
- Robust to individual model weaknesses

## 📁 Repository Structure


## 🚀 Installation

### Local Setup
```bash
# Clone repository
git clone https://github.com/navedjh/pneumonia-detection-task1.git
cd pneumonia-detection-task1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

#Train all models
python train.py



### Confusion Matrices

#### CNN Confusion Matrix
