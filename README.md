Neural Network Churn Prediction

1. Project Overview

This project explores the development and optimization of a feed-forward neural network designed to predict customer churn. The primary objective is to identify at-risk customers by analyzing demographic, contractual, and behavioral data.
To demonstrate fundamental deep learning principles, a custom neural network was implemented from scratch using NumPy, focusing on the mechanics of backpropagation and gradient descent. This implementation was then benchmarked against scikit-learn’s MLPClassifier to validate performance and assess generalization capabilities across various hyperparameter configurations.
The development environment utilized the following libraries:
NumPy: Essential for the low-level implementation of layer transformations and the optimization engine.
Pandas: Used for data manipulation and exploratory data analysis (EDA).
Matplotlib & Seaborn: Employed for generating training curves, statistical distributions, and confusion matrices.
Scikit-learn: Utilized for high-level data preprocessing, evaluation metrics, and comparative benchmarking experiments.

2. Dataset Characteristics

Dataset Composition
The analysis utilized a synthetic customer churn dataset consisting of 2,000 records. The predictive model utilizes categorical features—Region, Plan Type, Contract Type, and Payment Method—alongside numerical indicators to determine the target variable, Churn.
Class Distribution
A critical characteristic of this dataset is its severe class imbalance. The identified churn rate is approximately 3.2%, representing a 30:1 imbalance ratio. From an engineering perspective, this distribution significantly impacts model training; a naive model could achieve ~96.8% accuracy by simply predicting "No Churn" for every instance, necessitating a focus on minority-class precision and recall.
Statistical Summary
Exploratory Data Analysis (EDA) revealed distinct patterns between churning and non-churning customers. Evaluation of features such as tenure months, satisfaction scores, and monthly charges indicates that lower satisfaction scores and month-to-month contracts are high-impact indicators of churn risk.

3. Data Preprocessing Pipeline

To ensure robust training, a sequential preprocessing pipeline was established. In a custom NumPy implementation, gradient descent is highly sensitive to feature scales, making these steps non-negotiable for convergence:
Categorical Encoding: Features (Region, Plan Type, Contract Type, Payment Method) were transformed via LabelEncoder to provide the numerical inputs required for matrix operations.
Data Splitting: The dataset was partitioned into an 80/20 train-test split. Stratification was enforced to preserve the 30:1 churn ratio across both subsets, ensuring the test set remained a representative evaluation of the model's ability to detect the minority class.
Feature Scaling: StandardScaler was applied to normalize numerical features. Critically, the scaler was fitted only on the training data and subsequently applied to both sets. This approach prevents data leakage and ensures the model is evaluated on truly "unseen" data distributions.

4. Custom Neural Network Architecture

Model Structure
The NumPy implementation follows a feed-forward architecture with the following layer dimensions: [11 (Input Features)] → [32] → [16] → [1].
Mathematical Components
The network utilizes non-linear activation functions to map complex relationships between customer features and churn probability:
Layer
Function
Purpose
Hidden Layers
ReLU
Introduces non-linearity and mitigates vanishing gradients during backpropagation.
Output Layer
Sigmoid
Squashes outputs into a [0, 1] range for probabilistic interpretation.
Loss Function
Binary Cross-Entropy (BCE) was selected as the training objective. BCE is mathematically superior to Mean Squared Error (MSE) for this classification task because its logarithmic nature penalizes confident, incorrect predictions more heavily, aligning perfectly with the Sigmoid output for a probabilistic interpretation of churn.
Optimisation
The model utilizes mini-batch gradient descent with a learning rate of 0.005 and a batch size of 64.

5. Training and Evaluation Results

Training Performance
After 100 epochs, the custom neural network yielded the following performance metrics:
Final Training Accuracy: 100.00%
Final Test Accuracy: 96.50%
Model Assessment
Evaluation via the classification report and confusion matrix indicates that while the global accuracy is high, the model’s performance is heavily influenced by the majority class. The 100% training accuracy suggests the model successfully mapped the training distribution, but the test accuracy reflects the difficulty of generalizing the minority churn class under current constraints.
Visualisation Summary
The training curves recorded over 100 epochs indicate high convergence stability. Specifically, the loss curve shows a smooth, steady decrease, validating that the 0.005 learning rate provided an ideal balance between convergence speed and stability, avoiding the chaotic "overshooting" seen in higher-rate experiments.

6. Hyperparameter Experimentation

Empirical testing via MLPClassifier highlights the sensitivity of neural networks to structural and optimization parameters. Best generalization was achieved in configurations that prioritized network width or depth (Exps 2, 7) or increased stability through larger batch sizes (Exp 6).
Experiment
Configuration Change
Final Loss
Test Accuracy (%)
Result Summary
Exp 1
Baseline (32, 16), LR 0.005
0.0006
96.50%
Balanced baseline performance.
Exp 2
Deeper (64, 32, 16)
0.0001
97.25%
High generalization; superior feature extraction.
Exp 3
High LR (0.10)
0.0890
96.75%
Poor convergence; loss 148x worse than baseline.
Exp 4
Low LR (0.0005)
0.0190
96.75%
Slow convergence; remains under-trained.
Exp 5
Sigmoid Activation
0.0380
96.75%
Slower convergence than ReLU.
Exp 6
Large Batch (256)
0.0040
97.25%
Robust generalization through stable gradients.
Exp 7
Wider (128, 64)
0.0001
97.25%
Strongest performance; minimal train-test gap.

7. Theoretical Reflections

Role of Weights and Biases
Weights serve as the connection strengths that prioritize specific input features during the linear transformation Z=XW+b. Biases function as learnable offsets, providing the representational flexibility required for neurons to activate even when input signals are weak or zero.
Necessity of Non-linear Activation
Empirical results validate that non-linear activations are essential for churn prediction. Without ReLU and Sigmoid, the network would collapse into a single linear transformation, regardless of depth, rendering it incapable of learning the complex decision boundaries necessary to separate churning customers from the majority.
Learning Rate Impact
The experiments demonstrate a clear trade-off: a high learning rate (0.10) causes the optimizer to overshoot the global minimum, leading to a loss 148 times higher than the baseline. Conversely, a low learning rate (0.0005) results in an under-trained model that fails to reach peak performance within the allotted 100 epochs.
Generalisation Analysis
The gap between 100% training accuracy and 96.5% test accuracy in the baseline suggests that the model has effectively "memorized" the majority class. This is a nuance of imbalanced datasets; the high accuracy can be misleading, as it often reflects the model's proficiency at identifying "No Churn" cases while struggling with the 3.2% minority.

8. Key Limitations and Future Improvements

To improve the model's ability to identify the minority class (Churn), the following strategies are recommended:
Synthetic Minority Over-sampling Technique (SMOTE): To balance the 30:1 ratio by generating synthetic examples of the minority class.
Class-Weighted Loss Functions: Adjusting the BCE loss to apply a higher penalty for misclassifying churn events.
Advanced Metrics: Shifting evaluation focus from Accuracy to AUC-ROC and Precision-Recall curves.
Threshold Tuning: Optimizing the classification threshold to prioritize Recall, ensuring the model captures more at-risk customers even at the cost of slight increases in false positives.



