a. Problem Statement

The objective of this project is to predict whether an individual earns more than $50K per year using demographic and employment-related attributes from census data.

b. Dataset Description

The Adult Income dataset is sourced from the UCI Machine Learning Repository. It contains 48,842 records with 14 features including age, education, occupation, marital status, capital gain/loss, and hours worked per week.
The target variable is binary, indicating whether an individual earns more than $50K annually.

c. Models Used and Evaluation Metrics

| ML Model            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.8513   | 0.9043 | 0.7399    | 0.6005 | 0.6630 | 0.5740 |
| Decision Tree       | 0.8138   | 0.7445 | 0.6200    | 0.6087 | 0.6143 | 0.4916 |
| KNN                 | 0.7992   | 0.7273 | 0.5880    | 0.5869 | 0.5875 | 0.4548 |
| Naive Bayes         | 0.6167   | 0.8401 | 0.3811    | 0.9197 | 0.5389 | 0.3826 |
| Random Forest       | 0.8463   | 0.8784 | 0.7338    | 0.5792 | 0.6474 | 0.5572 |
| XGBoost             | 0.8736   | 0.9280 | 0.7831    | 0.6655 | 0.7195 | 0.6422 |

d. Model Observations

| ML Model            | Observation about Model Performance                                                                                                                                           |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Strong baseline model with high AUC (0.90) and balanced performance across metrics. Performs well on linearly separable patterns.                                             |
| Decision Tree       | Moderate accuracy but lower AUC, indicating overfitting and limited generalization capability.                                                                                |
| KNN                 | Lower performance due to high dimensionality after one-hot encoding. Sensitive to feature scaling and curse of dimensionality.                                                |
| Naive Bayes         | Very high recall (0.91) but low precision, meaning it predicts many positive cases but with higher false positives.                                                           |
| Random Forest       | Good overall performance and improved stability compared to Decision Tree. Handles feature interactions well.                                                                 |
| XGBoost             | Best performing model across almost all metrics. Highest Accuracy (0.8736), AUC (0.9280), F1 Score (0.7195), and MCC (0.6422). Demonstrates strong generalization capability. |
