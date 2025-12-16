# Model Improvement Report

## Executive Summary

### Key Achievements

- **Sensitivity Improvement**: +11.5 percentage points
- **False Negative Reduction**: -11.5 percentage points
- **Final Sensitivity**: 58.6%
- **Clinical Target Met**: ⚠️ Not Yet

## Methodology

### Implemented Improvements

1. **Focal Loss (γ=2.5)**
   - Automatically focuses on hard examples
   - Down-weights easy examples (confident correct predictions)
   - Up-weights misclassifications

2. **Label Smoothing (ε=0.1)**
   - Reduces overconfidence
   - Prevents model from outputting extreme probabilities
   - Improves calibration

3. **Tumor-Focused Augmentation**
   - Aggressive rotation (±54°)
   - Bilateral flips
   - Intensity variations

4. **Test Time Augmentation (Optional)**
   - Ensemble of 5 augmented predictions
   - Reduces variance
   - Improves robustness

## Detailed Metrics

| Metric | Base Model | Fine-Tuned | Focal Loss | Improvement |
|--------|------------|------------|------------|-------------|
| Recall | 47.1% | 74.7% | 58.6% | +11.5pp |
| Specificity | 100.0% | 75.0% | 98.9% | -1.1pp |
| Precision | 100.0% | 73.9% | 98.1% | -1.9pp |
| Accuracy | 73.4% | 74.9% | 79.3% | +5.9pp |

## Recommendations

### Further Improvements Needed

- Consider increasing Focal Loss gamma to 3.0
- Enable Test Time Augmentation (5-10 samples)
- Add more tumor samples via synthesis
- Ensemble multiple models
