# Model Improvement Report

## Executive Summary

### Key Achievements

- **Sensitivity Improvement**: +8.0 percentage points
- **False Negative Reduction**: -8.0 percentage points
- **Final Sensitivity**: 55.2%
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
| Recall | 47.1% | 75.9% | 55.2% | +8.0pp |
| Specificity | 100.0% | 78.3% | 100.0% | 0.0pp |
| Precision | 100.0% | 76.7% | 100.0% | 0.0pp |
| Accuracy | 73.4% | 77.1% | 78.2% | +4.8pp |

## Recommendations

### Further Improvements Needed

- Consider increasing Focal Loss gamma to 3.0
- Enable Test Time Augmentation (5-10 samples)
- Add more tumor samples via synthesis
- Ensemble multiple models
