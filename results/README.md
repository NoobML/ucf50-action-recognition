# 📊 Experimental Results

This directory contains all experimental results from training 7 sequence models on the UCF-50 dataset for video action recognition.

---

## 📁 Directory Structure
```
results/
├── plots/                          # Visualizations
│   ├── model_comparison.png       # 4-panel comprehensive comparison
│   ├── all_models_grid.png        # Training curves for all models
│   ├── GRU_curves.png             # GRU training curves (Winner)
│   ├── LSTM_curves.png            # LSTM training curves
│   ├── BiLSTM_curves.png          # BiLSTM training curves
│   ├── Simple_RNN_curves.png      # Simple RNN training curves
│   ├── Stacked_LSTM_curves.png    # Stacked LSTM training curves
│   ├── Transformer_curves.png     # Transformer training curves
│   └── Swin_Transformer_curves.png # Swin Transformer training curves
├── metrics/
│   └── training_results.json      # Complete training history
├── summary_report.txt             # Human-readable summary
└── README.md                      # This file
```

---

## 🏆 Summary Results

### Performance Ranking

| Rank | Model | Top-1 Acc (%) | Top-5 Acc (%) | Parameters (M) | Training Time (s) |
|------|-------|---------------|---------------|----------------|-------------------|
| 🥇 | **GRU** | **97.23** | **99.85** | **3.96** | **69.7** |
| 🥈 | LSTM | 96.63 | 99.93 | 5.27 | 81.2 |
| 🥉 | BiLSTM | 95.89 | 99.55 | 10.55 | 133.0 |
| 4 | Simple RNN | 94.99 | 99.33 | 1.34 | 57.6 |
| 5 | Swin Transformer | 93.79 | 99.18 | 4.23 | 70.9 |
| 6 | Stacked LSTM | 89.90 | 99.10 | 7.37 | 113.2 |
| 7 | Transformer | 3.14 | 11.37 | 13.68 | 161.9 |

---

## 📈 Key Findings

### 🏆 Best Overall Model: GRU

**Why GRU Won:**
- Highest Top-1 accuracy: 97.23%
- Near-perfect Top-5 accuracy: 99.85%
- Excellent parameter efficiency: 3.96M parameters
- Fast training: 69.7 seconds
- Optimal balance of capacity and generalization

### ⚡ Most Efficient: Simple RNN

**Efficiency Score: 70.89** (accuracy per million parameters)
- 94.99% accuracy with only 1.34M parameters
- Fastest training: 57.6 seconds
- Best for edge devices and real-time applications

### 🚫 Notable Failure: Transformer

- Catastrophic failure: 3.14% accuracy (barely above random)
- Highest parameter count: 13.68M
- Longest training time: 161.9 seconds
- **Lesson:** Transformers require large datasets to perform well

---

## 📊 Visualizations Guide

### 1. `model_comparison.png`
**4-Panel Comprehensive Comparison**

This is the **main visualization** showing:
- **Top-Left:** Top-1 Accuracy comparison (horizontal bar chart)
- **Top-Right:** Top-5 Accuracy comparison (horizontal bar chart)
- **Bottom-Left:** Training time comparison (horizontal bar chart)
- **Bottom-Right:** Model size comparison (horizontal bar chart)

**Use this for:**
- Quick overview of all models
- Presentations and papers
- LinkedIn posts

---

### 2. `all_models_grid.png`
**Training Dynamics - All Models**

Shows training and validation curves (loss + accuracy) for all 7 models in a grid layout.

**Observations:**
- GRU/LSTM/Simple RNN: Smooth, stable convergence
- BiLSTM: Similar to LSTM but slightly lower final accuracy
- Stacked LSTM: Shows overfitting (larger train-val gap)
- Transformer: Erratic, non-convergent behavior
- Swin Transformer: Stable but lower accuracy than recurrent models

**Use this for:**
- Comparing training stability across models
- Identifying overfitting patterns
- Demonstrating convergence behavior

---

### 3. Individual Model Curves

Each model has its own training curve plot showing:
- **Left panel:** Training and validation loss over epochs
- **Right panel:** Training and validation accuracy over epochs

#### `GRU_curves.png` 🏆
- Fastest convergence rate
- Reaches >95% by epoch 7
- Minimal train-val gap (excellent generalization)
- Smooth, monotonic improvement

#### `LSTM_curves.png`
- Very similar to GRU
- Slightly slower convergence
- Excellent stability
- Final accuracy: 96.63%

#### `BiLSTM_curves.png`
- Bidirectional processing visible in slower initial learning
- Final accuracy: 95.89% (lower than unidirectional LSTM)
- Doubled parameters didn't improve performance

#### `Simple_RNN_curves.png`
- Rapid early convergence
- Slight overfitting visible (5% train-val gap)
- Remarkable 94.99% accuracy for simplest architecture

#### `Stacked_LSTM_curves.png`
- Shows clear overfitting (training acc ~95%, validation ~90%)
- Depth didn't help on this dataset
- Suggests dataset complexity doesn't require hierarchical learning

#### `Transformer_curves.png`
- **Complete failure to learn**
- Loss barely decreases
- Accuracy stays near random (3.14% vs 2% random)
- Demonstrates importance of dataset scale for attention mechanisms

#### `Swin_Transformer_curves.png`
- More stable than standard Transformer
- 93.79% accuracy shows attention can work with constraints
- Window-based attention provides useful inductive bias

---

## 📄 Files Description

### `training_results.json`

Complete training history in JSON format containing:
```json
{
  "model_name": {
    "train_losses": [...],      // Loss per epoch (training)
    "val_losses": [...],         // Loss per epoch (validation)
    "train_accs": [...],         // Accuracy per epoch (training)
    "val_accs": [...],           // Accuracy per epoch (validation)
    "learning_rates": [...],     // Learning rate per epoch
    "training_time": X.X,        // Total training time (seconds)
    "best_val_acc": XX.XX,       // Best validation accuracy achieved
    "top5_acc": XX.XX,           // Top-5 accuracy on validation set
    "total_params": XXXXXXX      // Number of trainable parameters
  }
}
```

**Use this for:**
- Reproducing plots
- Detailed analysis
- Comparing specific epochs
- Research paper data

---

### `summary_report.txt`

Human-readable text summary containing:
- Best overall model
- Fastest training model
- Most efficient model (accuracy/parameters)
- Performance statistics
- Project metadata (total videos, training time, etc.)

---

## 🔬 Experimental Setup

### Dataset
- **Name:** UCF-50 (University of Central Florida Action Recognition)
- **Total Videos:** 6,681
- **Classes:** 50 human action categories
- **Train Split:** 5,344 videos (80%)
- **Validation Split:** 1,337 videos (20%)

### Feature Extraction
- **Backbone:** ResNet50 (pretrained on ImageNet)
- **Frames per Video:** 32 (uniformly sampled)
- **Feature Dimension:** 2048
- **Total Features Extracted:** 1.63 GB

### Training Configuration
```python
BATCH_SIZE = 16
LEARNING_RATE = 0.001
OPTIMIZER = Adam
WEIGHT_DECAY = 1e-5
EPOCHS = 20
SCHEDULER = StepLR (step_size=5, gamma=0.5)
DROPOUT = 0.3
HIDDEN_DIM = 512
```

### Hardware
- **Platform:** Kaggle Notebooks
- **GPU:** Tesla T4 (16GB VRAM)
- **Training Time (all models):** 11.5 minutes
- **Feature Extraction Time:** ~15 minutes (one-time)

---

## 📈 Statistical Analysis

### Performance Distribution
```
Mean Accuracy: 81.37%
Median Accuracy: 94.99%
Std Dev: 32.77%
Min: 3.14% (Transformer)
Max: 97.23% (GRU)
Range: 94.09%
```

**Note:** Statistics heavily skewed by Transformer's failure. Excluding Transformer:
```
Mean Accuracy: 94.73%
Median Accuracy: 94.99%
Std Dev: 2.53%
Min: 89.90% (Stacked LSTM)
Max: 97.23% (GRU)
Range: 7.33%
```

### Parameter Efficiency

**Efficiency Score = (Top-1 Accuracy) / (Parameters in millions)**
```
Simple RNN:    70.89 (most efficient)
GRU:           24.55
Swin Trans:    22.18
LSTM:          18.34
Stacked LSTM:  12.20
BiLSTM:         9.09
Transformer:    0.23 (least efficient)
```

### Training Speed

**Seconds per percentage point of accuracy:**
```
Simple RNN:    0.61 sec/acc%
GRU:           0.72 sec/acc%
Swin Trans:    0.76 sec/acc%
LSTM:          0.84 sec/acc%
BiLSTM:        1.39 sec/acc%
Stacked LSTM:  1.26 sec/acc%
Transformer:  51.56 sec/acc% (extremely inefficient)
```

---

## 🎯 Model Selection Guide

### For Maximum Accuracy
```
Recommended: GRU
Accuracy: 97.23%
Use Case: Production systems, research benchmarks
```

### For Edge Devices / Real-Time
```
Recommended: Simple RNN
Accuracy: 94.99%
Parameters: 1.34M
Use Case: Mobile apps, IoT, embedded systems
```

### For Balanced Performance
```
Recommended: LSTM
Accuracy: 96.63%
Parameters: 5.27M
Use Case: General video understanding tasks
```

### What to Avoid
```
NOT Recommended: Transformer (catastrophic failure)
NOT Recommended: Stacked LSTM (overengineered, underperformed)
```

---

## 📝 Reproducing Results

### Using the Notebook
```bash
# Open Kaggle notebook
cd notebook/
jupyter notebook ucf50_sequence_models.ipynb

# Or view on Kaggle
# https://www.kaggle.com/YOUR_USERNAME/ucf50-sequence-models
```

### Using Training Scripts
```bash
# Train specific model
python scripts/train.py --model GRU --epochs 20

# Train all models
python scripts/train.py --all

# Generate visualizations
python scripts/visualize.py --results results/metrics/training_results.json
```

---

## 📚 Citation

If you use these results in your research, please cite:
```bibtex
@techreport{saeed2025ucf50,
  title={Comparative Analysis of Sequence Models for Video Action Recognition on UCF-50},
  author={Saeed, Mushtaq},
  year={2025},
  institution={CECOS University},
  note={Experimental results: GRU achieved 97.23\% accuracy}
}
```

---

## 📧 Questions?

For questions about these results:
- **Email:** mushtaqsaeed577@gmail.com
- **GitHub Issues:** [Open an issue](../../issues)
- **LinkedIn:** [Your Profile](#)

---

## 🔄 Last Updated

**Date:** February 2, 2025  
**Experiment ID:** UCF50-SeqModels-v1  
**Status:** Complete ✅

---

**All visualizations are publication-quality (300 DPI) and ready for use in papers, presentations, and reports!**