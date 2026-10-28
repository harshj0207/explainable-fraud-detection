# Explainable Credit Card Fraud Detection Using Deep Learning Ensembles

A research-based implementation of an explainable credit-card fraud detection framework inspired by an attention-based ensemble of CNN, RNN, LSTM, and GNN models with DOWA/IOWA fusion and confidence-aware selection.

## Project Overview

Credit-card fraud detection is challenging because fraudulent transactions are rare and false alarms can inconvenience legitimate customers. This project implements a multi-model deep learning pipeline and evaluates its performance under severe class imbalance.

The implementation is inspired by:

**Paper:** *Explainable AI for Fraud Detection: An Attention-Based Ensemble of CNNs, GNNs, and A Confidence-Driven Gating Mechanism*

## Objectives

- Implement four base model families: CNN, RNN, LSTM, and GNN.
- Combine model predictions using simplified DOWA and IOWA fusion mechanisms.
- Select a fused output using a confidence-based rule.
- Use an MLP meta-learner for final classification.
- Apply SHAP to investigate influential input features.
- Evaluate the system using fraud-relevant metrics and threshold tuning.

## Tech Stack

- Python
- PyTorch
- Pandas and NumPy
- Scikit-learn
- SHAP
- Matplotlib and Seaborn

## Dataset

The project uses the Kaggle Credit Card Fraud Detection dataset, containing 284,807 transactions and 492 fraud cases.

Dataset source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset is highly imbalanced. The notebook uses the paper's reported top-ten feature ranking as the starting point for feature selection.

## Methodology

1. Load and inspect the transaction data.
2. Split the data into training, validation, and test sets.
3. Scale the features using training-set statistics.
4. Train CNN, RNN, LSTM, and GNN base models with class-weighted loss.
5. Fuse CNN and RNN outputs using a simplified DOWA approach.
6. Fuse GNN and LSTM outputs using learnable IOWA-style weights.
7. Select a fused output using a confidence proxy.
8. Train an MLP meta-learner and evaluate the final model.
9. Use SHAP to examine feature contributions.
10. Tune the classification threshold on validation data.

## Results

### Default threshold: 0.5

| Metric | Ensemble result |
|---|---:|
| Precision | 0.0591 |
| Recall | 0.8878 |
| F1-score | 0.1108 |
| ROC-AUC | 0.9731 |
| PR-AUC | 0.6339 |

### Validation-selected threshold: approximately 0.9554

| Metric | Ensemble result |
|---|---:|
| Precision | 0.8452 |
| Recall | 0.7245 |
| F1-score | 0.7802 |
| False positives | 13 |
| Fraud cases detected | 71 out of 98 |

Threshold selection improves precision and F1-score on the test set in this experiment, while recall decreases. The threshold was selected using validation data rather than the test labels.

## Important Implementation Notes

This is a practical implementation inspired by the research paper, not an exact reproduction of every experimental detail.

- The base models use reduced training durations to fit available compute resources.
- DOWA is implemented using a simplified weighting approach.
- The IOWA-style weights are trained using a simplified objective.
- The GNN uses a feature-space nearest-neighbour graph constructed within batches rather than a known transaction network.
- The meta-learner uses base-model predictions generated in the implementation; out-of-fold predictions would provide a more rigorous stacking evaluation.
- SHAP is applied to the trained implementation and does not constitute an exact reproduction of the paper's original feature-selection experiment.

These differences should be considered when comparing results with the original paper.

## How to Run

1. Download the dataset from the source linked above.
2. Open `fraud_detection_ensemble.ipynb` in Google Colab or Kaggle Notebooks.
3. Configure the dataset path for your environment.
4. Install any missing dependencies.
5. Run the notebook cells in order.

## Future Improvements

- Use out-of-fold predictions for stacking.
- Add component-wise ablation experiments.
- Evaluate statistical uncertainty using repeated runs and confidence intervals.
- Improve graph construction and confidence calibration.
- Compare against traditional baselines such as Logistic Regression, Random Forest, and XGBoost.

## References

Chagahi, M. H., Delfan, N., Dashtaki, S. M., Moshiri, B., and Piran, M. J. *Explainable AI for Fraud Detection: An Attention-Based Ensemble of CNNs, GNNs, and A Confidence-Driven Gating Mechanism*. arXiv:2410.09069, 2025.

Paper: https://arxiv.org/abs/2410.09069