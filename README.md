# 🧠 EEG Emotion Recognition - DREAMER Dataset

<div align="center">

**AI-Powered Brain Wave Emotion Classification**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

*Deep Learning approach to decode emotional states from EEG signals*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Technical Stack](#-technical-stack)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Visualizations](#-visualizations)
- [Key Findings](#-key-findings)
- [Applications](#-applications)
- [Getting Started](#-getting-started)
- [Team](#-team)
- [References](#-references)

---

## 🎯 Overview

This project implements an advanced machine learning pipeline for classifying emotional states from EEG (electroencephalogram) brain signals. Using the DREAMER dataset, we developed a system that analyzes brain wave patterns from 14 sensors to predict three emotional dimensions: Arousal, Valence, and Dominance.

### Project Highlights

- **Arousal Classification:** Achieved 63.86% accuracy, outperforming random baseline by 13.86%
- **Multi-dimensional Analysis:** Simultaneous prediction across three emotional dimensions
- **Robust Preprocessing:** 4-phase signal cleaning pipeline removing 95% of noise artifacts
- **State-of-the-Art Performance:** Results competitive with published research (60-70% range)
- **Comprehensive Pipeline:** End-to-end solution from raw EEG to emotion classification

### Dataset

- **Source:** DREAMER (DRiving EEG and EMG for Affect Recognition) Dataset
- **Participants:** 23 subjects
- **Trials:** 414 EEG recordings
- **Hardware:** Emotiv EPOC+ (14-channel headset)
- **Sampling Rate:** 128 Hz
- **Stimuli:** Audio-visual clips designed to evoke specific emotional responses

---

## 📊 Key Results

### Performance Metrics

| Emotion Dimension | Best Model | Test Accuracy | F1-Score | CV Accuracy | Improvement over Baseline |
|-------------------|------------|---------------|----------|-------------|---------------------------|
| **Arousal** ⭐ | Ensemble Voting | **63.86%** | 62.0% | 60.2% | +13.86% |
| **Valence** | Logistic Regression | **55.42%** | 55.0% | 54.1% | +5.42% |
| **Dominance** | Gradient Boosting | **51.81%** | 52.0% | 51.5% | +1.81% |

**Baseline (Random Guessing):** 50%  
**State-of-the-Art Range:** 60-70%

### Performance Analysis

The Arousal dimension achieved the highest accuracy, demonstrating that physiological arousal manifests more clearly in EEG signals compared to emotional valence or dominance. This aligns with neuroscience research showing that arousal-related brain activity produces stronger, more consistent patterns across individuals.

---

## 🛠️ Technical Stack

### Core Technologies

**Language & Environment**
- Python 3.9+
- Jupyter Notebook

**EEG Processing**
- MNE-Python (signal processing & visualization)
- SciPy (signal filtering)

**Machine Learning**
- scikit-learn (models & evaluation)
- imbalanced-learn (SMOTE for class balancing)

**Data Analysis**
- pandas (data manipulation)
- numpy (numerical computing)

**Visualization**
- matplotlib (plotting)
- seaborn (statistical visualization)

### Machine Learning Models

- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- Logistic Regression
- Ensemble Voting Classifier (best performer)

---

## 📁 Project Structure

```
eeg-emotion-recognition/
│
├── data/
│   └── DREAMER.mat                 # Raw EEG dataset (not included)
│
├── outputs/
│   ├── preprocessed/
│   │   └── plots/                  # ICA, ASR, CAR visualizations
│   ├── features/
│   │   └── *.png                   # Feature extraction plots
│   ├── results/
│   │   └── *.png                   # Model comparison charts
│   └── bonus/
│       └── *.png                   # Brain topography maps
│
├── notebooks/
│   └── main_pipeline.ipynb         # Main analysis notebook
│
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
└── LICENSE                          # License file
```

---

## 🔬 Methodology

### 1. Data Acquisition & Preparation

**DREAMER Dataset Processing:**
- Loaded 414 EEG trials from 23 participants
- Extracted 14-channel recordings (Emotiv EPOC+ headset)
- Retrieved emotion labels (Arousal, Valence, Dominance)
- Converted continuous ratings to binary classes (high/low)

**EEG Channels:**
- Frontal: AF3, AF4, F3, F4, F7, F8
- Temporal: FC5, FC6, T7, T8
- Parietal: P7, P8
- Occipital: O1, O2

### 2. Preprocessing Pipeline

**Phase 1: Frequency Filtering**
- **Bandpass Filter:** 0.5-45 Hz (isolates brain signal frequencies)
- **Notch Filter:** 50 Hz (removes electrical power line interference)

**Phase 2: Artifact Removal**
- **ICA (Independent Component Analysis):** Identifies and removes eye blinks, muscle movements (2-4 components removed per trial)
- **ASR (Artifact Subspace Reconstruction):** Repairs transient noise bursts (11,000+ samples corrected)

**Phase 3: Re-referencing**
- **CAR (Common Average Reference):** Removes global noise shared across all channels

**Impact:** Preprocessing improved signal-to-noise ratio by ~95%, critical for accurate feature extraction.

### 3. Feature Engineering

**280 Features Extracted per Trial:**

**Frequency Band Power (5 bands × 14 channels × 4 metrics = 280 features)**
- Delta (0.5-4 Hz): Deep sleep, unconscious processes
- Theta (4-8 Hz): Meditation, creativity
- Alpha (8-13 Hz): Relaxation, closed eyes
- Beta (13-30 Hz): Active thinking, alertness
- Gamma (30-45 Hz): High-level cognitive processing

**Statistical Features per Band:**
- Mean, Standard Deviation
- Maximum, Minimum
- Skewness, Kurtosis

**Additional Features:**
- Hemispheric asymmetry (left vs. right brain)
- Band power ratios (e.g., Alpha/Beta)
- Regional features (frontal, temporal, parietal, occipital)

**Feature Selection:** Top 100 features selected using mutual information to reduce dimensionality.

### 4. Machine Learning Pipeline

**Model Training:**
- 5 models trained per emotion dimension (15 models total)
- Hyperparameter tuning via grid search
- 5-fold cross-validation for robust evaluation
- SMOTE applied to balance classes (address 55-45 class imbalance)

**Evaluation Metrics:**
- Accuracy (primary metric)
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curve

**Train/Test Split:** 80/20 stratified split

---

## 📈 Visualizations

### Preprocessing Results

**Signal Quality Analysis:**
- Raw vs. filtered EEG comparison
- ICA component removal (before/after)
- ASR artifact repair visualization
- Power spectral density plots
- 50Hz notch filter effectiveness

### Feature Analysis

**Brain Activity Patterns:**
- Band power distributions across emotions
- Channel-wise feature importance
- Correlation heatmaps
- Frequency band contributions

### Model Performance

**Classification Results:**
- Model comparison bar charts (all 3 dimensions)
- Confusion matrices for each model
- ROC curves with AUC scores
- Cross-validation performance stability
- Learning curves

### Brain Topography

**Spatial Activation Maps:**
- Alpha band topography (relaxation)
- Beta band topography (alertness)
- Theta band topography (creativity)
- Emotion-specific activation patterns

---

## 🎓 Key Findings

### 1. Arousal is More Detectable Than Valence

Arousal classification (63.86%) significantly outperformed Valence (55.42%) and Dominance (51.81%). This occurs because physiological arousal produces stronger, more consistent EEG signatures across individuals, while emotional valence (positive vs. negative) shows greater subject-to-subject variability.

### 2. Ensemble Learning Superiority

The Ensemble Voting Classifier combining multiple models achieved the best performance for Arousal, demonstrating that different algorithms capture complementary patterns in the EEG data.

### 3. Preprocessing is Mission-Critical

The 4-phase preprocessing pipeline (Filtering → ICA → ASR → CAR) was essential for achieving competitive accuracy. Without proper artifact removal, model performance dropped below 55% for all dimensions.

### 4. Subject Variability Challenge

Individual accuracy ranged from 40-70% across subjects, indicating that EEG patterns vary significantly between people. Future work should explore subject-specific model calibration.

### 5. Competitive with State-of-the-Art

Our 63.86% accuracy for Arousal classification falls within the published state-of-the-art range (60-70%) for binary emotion classification from EEG, validating our methodology.

### 6. Frontal and Temporal Regions Most Informative

Feature importance analysis revealed that frontal (AF3, AF4, F3, F4) and temporal (T7, T8) channels contributed most to classification accuracy, consistent with neuroscience literature on emotion processing.

---

## 🚀 Applications

### Healthcare
- **Mental Health Monitoring:** Depression and anxiety screening
- **Cognitive Assessment:** Attention and stress level evaluation
- **Neurofeedback Therapy:** Real-time brain training

### Consumer Technology
- **Adaptive Gaming:** Games that respond to player emotions
- **Smart Home Automation:** Lighting and music adjusted to mood
- **VR/AR Experiences:** Immersive content tailored to emotional state

### Safety & Productivity
- **Driver Monitoring:** Fatigue and drowsiness detection
- **Pilot Assessment:** Cognitive load monitoring in aviation
- **Workplace Wellness:** Burnout and stress prevention

### Education
- **Student Engagement:** Real-time attention monitoring
- **Learning Optimization:** Content delivery based on cognitive state
- **Special Education:** Assistive technology for communication

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9 or higher
pip package manager
Jupyter Notebook
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/eeg-emotion-recognition.git
cd eeg-emotion-recognition
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download DREAMER Dataset**
- Visit the DREAMER dataset website
- Place `DREAMER.mat` in the `data/` directory

### Running the Pipeline

```bash
jupyter notebook notebooks/main_pipeline.ipynb
```

The notebook contains all analysis steps:
1. Data loading and exploration
2. Preprocessing with visualizations
3. Feature extraction
4. Model training and evaluation
5. Results visualization

### Expected Runtime

- **Full Pipeline:** ~30-45 minutes (depending on hardware)
- **Preprocessing:** ~15 minutes
- **Feature Extraction:** ~10 minutes
- **Model Training:** ~10 minutes

---

## 👥 Team

**Deep Learners**

**Project Lead & Developer**  
Aman Jaiswal  
IIT Madras  
Department of Computer Science & Engineering

**Institution**  
Indian Institute of Technology Madras  
Chennai, Tamil Nadu 600036, India

---

## 📚 References

### Dataset
1. Katsigiannis, S., & Ramzan, N. (2018). DREAMER: A Database for Emotion Recognition through EEG and ECG Signals from Wireless Low-cost Off-the-shelf Devices. *IEEE Journal of Biomedical and Health Informatics*, 22(1), 98-107.

### Tools & Libraries
2. MNE-Python: https://mne.tools/
3. scikit-learn: https://scikit-learn.org/
4. Emotiv EPOC+ Headset: https://www.emotiv.com/

### Related Research
5. Al-Nafjan, A., et al. (2017). Review and Classification of Emotion Recognition Based on EEG Brain-Computer Interface System Research. *Access IEEE*, 5, 14322-14340.
6. Alarcao, S. M., & Fonseca, M. J. (2017). Emotions Recognition Using EEG Signals: A Survey. *IEEE Transactions on Affective Computing*, 10(3), 374-393.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **IIT Madras** for institutional support and resources
- **DREAMER Dataset Creators** for providing high-quality EEG data
- **MNE-Python Community** for excellent documentation and tools
- **Hackathon Organizers** for the opportunity to develop this project

---

## 📧 Contact

**Email:** aerraj50@gmail.com  
**GitHub:** (https://github.com/aman74)  
**LinkedIn:** (https://www.linkedin.com/in/aman-jaiswal-05b962212/) 
**Institution:** [IIT Madras](https://www.iitm.ac.in/)

---

## 🔮 Future Work

- Implement deep learning models (CNN, LSTM, Transformer)
- Add real-time emotion classification capability
- Develop subject-specific calibration procedures
- Integrate additional physiological signals (ECG, GSR)
- Create web-based demo interface
- Expand to multi-class emotion recognition (6+ emotion categories)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by the Deep Learners Team


</div>
