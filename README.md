# cnn_food

# CNN Model for Delivery Classification

## Overview
This project implements a Convolutional Neural Network (CNN) to classify deliveries as "Fast" or "Delayed" based on image data. The model is trained using 5-fold cross-validation and achieves an accuracy of 1.0000.

## Methodology
1. **Data Preparation:**
   - Images representing delivery locations are generated.
   - Labels are assigned based on delivery times (Fast or Delayed).
   - Images are preprocessed (grayscale conversion, resizing, and normalization).

2. **Model Architecture:**
   - Input Layer: (64x64x1) grayscale image
   - Conv2D (32 filters, 3x3, ReLU) + MaxPooling (2x2)
   - Conv2D (64 filters, 3x3, ReLU) + MaxPooling (2x2)
   - Conv2D (128 filters, 3x3, ReLU)
   - Flatten layer
   - Dense (128 neurons, ReLU)
   - Output Layer: Dense (2 neurons, Softmax)

3. **Training Process:**
   - 5-Fold Cross-Validation
   - Categorical Crossentropy Loss Function
   - Adam Optimizer
   - 5 Epochs per fold
   - Batch size: 32

## Model Performance
- **Validation Accuracy across all folds:** 1.0000
- **Final Evaluation:**
  - The model perfectly classifies all test images.
  - No loss reported after the first epoch, indicating strong generalization.

## Key Findings
- The model achieves perfect accuracy due to a well-structured dataset.
- Further testing on unseen real-world data is recommended to confirm robustness.
- The small dataset might contribute to overfitting, requiring augmentation strategies.

## Usage
1. Place images in the `images/` directory.
2. Run `train.py` to train the model.
3. Predictions can be made using `predict.py`.

## Dependencies
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Future Improvements
- Introduce real-world images for validation.
- Use data augmentation to prevent overfitting.
- Test the model with different architectures for efficiency.

