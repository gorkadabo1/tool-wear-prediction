# Tool Wear Prediction with Deep Learning

Binary classification system to predict tool wear condition in CNC machining processes using a hybrid CNN-LSTM neural network architecture.

## Overview

This project addresses a critical challenge in industrial manufacturing: predicting when cutting tools become worn during CNC machining operations. Early detection of tool wear can prevent defective products, reduce downtime, and optimize maintenance schedules.

The solution implements a hybrid deep learning model that combines Convolutional Neural Networks (CNN) for local feature extraction with Bidirectional Long Short-Term Memory (LSTM) networks for capturing temporal dependencies in time-series sensor data.

## Architecture

The `HybridCNNRNN` model consists of three main components:

**Convolutional Layers**
- Conv1D (47 -> 32 channels, kernel=5) with batch normalization
- Conv1D (32 -> 64 channels, kernel=3) with batch normalization
- MaxPooling and Dropout (0.3) after each block

**Recurrent Layers**
- Bidirectional LSTM (2 layers, 128 hidden units per direction)
- Total output: 256 features (128 x 2 directions)

**Fully Connected Layers**
- Dense (256 -> 64) with ReLU and Dropout
- Dense (64 -> 1) with Sigmoid activation for binary classification

## Dataset

The project uses the [CNC Mill Tool Wear dataset](https://www.kaggle.com/shasun/tool-wear-detection-in-cnc-mill) from the University of Michigan, containing data from 18 machining experiments on wax blocks with various speed and pressure configurations.

Each experiment includes:
- Position data (X, Y, Z axes)
- Spindle rotation measurements
- Feed rate readings
- Machining process indicators

Recordings are sampled at 100ms intervals, with binary labels indicating worn or unworn tool condition.

## Methodology

### Data Processing

1. **Sliding Window Approach**: Time series data is segmented into fixed-size windows (128 samples) with configurable step size (10 samples) for training
2. **Normalization**: StandardScaler applied to all features
3. **Train/Val/Test Split**: Stratified split ensuring both classes are represented in each set

### Training

- Optimizer: Adam (lr=5e-4, weight_decay=1e-5)
- Loss: Binary Cross-Entropy
- Model selection based on best validation F1-score
- GPU acceleration supported (CUDA)

### Evaluation

- **Window-level metrics**: Direct predictions on individual time windows
- **Experiment-level metrics**: Majority voting aggregates window predictions per experiment

### Alternative: Leave-One-Out Cross-Validation

A variant implementation (`HybridCNNRNNNoWindow`) processes complete experiment sequences without windowing, evaluated using LOOCV for robust performance estimation with limited data.

## Results

### Main Model (HybridCNNRNN with Sliding Windows)

| Level | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Window | 41.38% | - | 85.71% | 55.81% |
| Experiment | 75.00% | 66.67% | 100.00% | **80.00%** |

The experiment-level F1-score of 80% demonstrates that majority voting effectively compensates for individual window prediction errors.

### LOOCV Variant

- Average F1-Score: 32.36%
- Average Accuracy: 43.76%
- High variance across experiments (F1 range: 0.00 - 0.98)

The LOOCV results highlight the challenge of generalizing with only 18 experiments and suggest the sliding window approach provides better regularization.

## Project Structure

```
tool-wear-prediction/
├── README.md
├── src/
│   └── tool_wear_prediction.py
├── docs/
│   └── report.pdf
└── data/
    └── README.md
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- scikit-learn
- gdown (for dataset download)

## Usage

```bash
# Clone the repository
git clone https://github.com/gorkadabo1/tool-wear-prediction.git
cd tool-wear-prediction

# Install dependencies
pip install torch numpy pandas scikit-learn gdown

# Run main training and evaluation
python src/tool_wear_prediction.py

# Run LOOCV evaluation (alternative)
# Uncomment the loocv_evaluation() call in the script
```

## Key Techniques

- Hybrid CNN-LSTM architecture for spatiotemporal feature learning
- Sliding window segmentation for time series data augmentation
- Bidirectional LSTM for capturing both past and future context
- Majority voting ensemble for robust experiment-level predictions
- Leave-One-Out Cross-Validation for small dataset evaluation
- Batch normalization and dropout for regularization

## Future Work

- Implement attention mechanisms for improved feature weighting
- Explore data augmentation techniques for time series
- Test with larger industrial datasets
- Add real-time prediction capabilities
- Investigate transfer learning from related domains

## License

This project is available for academic and educational purposes at the University of the Basque Country (EHU).
