# Interpretable Heart Disease Classification

### A Comparative Study of Six Machine Learning Classifiers Using SMOTETomek and SHAP

This repository contains the implementation and research paper for a machine learning study on heart disease classification using the 2023 Behavioral Risk Factor Surveillance System (BRFSS) dataset.

## Research Objective

The study investigates the performance of multiple machine learning classifiers for heart disease classification while addressing class imbalance and model interpretability.

The work focuses on:

- Comparative evaluation of six machine learning classifiers
- Handling severe class imbalance using SMOTETomek
- Evaluation using multiple performance metrics rather than accuracy alone
- Model interpretation using SHAP (SHapley Additive exPlanations)
- Maintaining a leakage-aware train/test evaluation workflow

## Dataset

The study uses the 2023 BRFSS combined landline and cellular telephone dataset (LLCP2023), published by the Centers for Disease Control and Prevention (CDC).

The raw dataset contains 433,323 records and 350 variables. After removing records with a missing target label, 428,738 records remained.

For reproducible experimentation, the cleaned data was downsampled to 80,000 records using `random_state = 42`.

The resulting dataset contained:

- 73,280 negative cases
- 6,720 positive cases
- Positive class proportion: 8.4%

The target variable `_MICHD` represents self-reported history of coronary heart disease (CHD) or myocardial infarction (MI).

> The dataset represents self-reported survey responses. Survey weights were not applied, so the results should not be interpreted as population-level prevalence estimates.

## Methodology

The experimental pipeline consists of:

1. Data cleaning and duplicate-column removal
2. Selection of 26 demographic, behavioral, health-related, and socioeconomic predictors
3. Stratified 80/20 train-test split
4. Missing-value imputation using the most-frequent strategy
5. Feature scaling using StandardScaler
6. Class balancing using SMOTETomek on the training data only
7. Training and comparison of six classification approaches
8. Evaluation on an untouched imbalanced test set
9. Five-fold stratified cross-validation on the training partition
10. SHAP-based model interpretation

### Class Imbalance

The original dataset is strongly imbalanced toward the negative class.

SMOTETomek was applied only to the training data. SMOTE generates synthetic minority-class samples, while Tomek-link undersampling removes ambiguous majority-class samples near class boundaries.

The test set was kept untouched to provide an unbiased evaluation of model performance.

## Models Evaluated

The study compares:

- Majority-class baseline
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (RBF)
- XGBoost

## Evaluation Metrics

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Particular attention was given to minority-class recall and F1-score because accuracy alone can be misleading on imbalanced health-related classification problems.

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority Baseline | 91.60% | 0.000 | 0.000 | 0.000 | — |
| Logistic Regression | 74.05% | 0.214 | 0.780 | 0.336 | 0.835 |
| Decision Tree | 81.55% | 0.247 | 0.583 | 0.347 | 0.803 |
| Random Forest | 83.67% | 0.276 | 0.583 | 0.375 | 0.839 |
| SVM (RBF) | 77.76% | 0.229 | 0.698 | 0.345 | 0.820 |
| XGBoost | 91.51% | 0.480 | 0.122 | 0.195 | 0.845 |

XGBoost achieved the highest overall accuracy and ROC-AUC among the evaluated models, while Random Forest achieved the highest F1-score.

These results demonstrate that higher accuracy does not necessarily correspond to better minority-class detection.

## Model Explainability

SHAP was used to examine the contribution of individual features to XGBoost predictions.

Among the most influential features identified were:

- Age group
- General health
- High blood pressure
- Income
- High cholesterol
- Sex

SHAP values describe the contribution of features to model predictions and should not be interpreted as evidence of causal relationships.

## Reproducibility

The main implementation notebook is available at:

`code/Heart_Disease_Classification_Final.ipynb`

The research paper is available in the root directory:

`Interpretable_Heart_Disease_Classification_Using_SMOTETomek_and_SHAP.pdf`

The experiments use fixed random seeds where applicable to support reproducibility.

## Limitations

This study has several limitations:

- The target represents self-reported heart disease history rather than clinical diagnosis.
- BRFSS survey weights were not applied.
- The experiments use a reproducible 80,000-record sample rather than the full cleaned dataset.
- Model performance depends on the selected features, preprocessing strategy, and classification threshold.
- SHAP explains model behavior but does not establish causal relationships.
- Further work could investigate threshold optimization and cost-sensitive learning for improved minority-class detection.

## Tools and Libraries

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- imbalanced-learn
- SHAP
- Matplotlib
- Seaborn
- Google Colab

## Repository Contents

```text
Interpretable-Heart-Disease-Classification/
│
├── README.md
├── Interpretable_Heart_Disease_Classification.pdf
│
└── code/
    └── Heart_Disease_Classification_Final.ipynb

```markdown
## Author

**Kausturi Chakraborty**  
Department of Computer Science and Engineering  
Adamas University, Kolkata, India
