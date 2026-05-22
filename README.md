# Laptop Price Prediction using Machine Learning

## Project Description

This project builds a complete machine learning regression pipeline for predicting laptop prices based on hardware specifications and laptop characteristics. Multiple regression algorithms were implemented, compared, tuned, and evaluated using cross-validation and learning curves.

The project focuses on:

* Data preprocessing
* Feature engineering
* Model comparison
* Hyperparameter tuning
* Cross-validation
* Error analysis
* Performance evaluation

---

## Dataset Features

The dataset includes laptop specifications such as:

* Company
* Product Name
* Type Name
* RAM
* CPU Company
* GPU Company
* Operating System
* Memory Type
* Memory Size
* Screen Resolution
* Screen Width & Height
* Weight
* Price (Euro)

Target variable:

* `Price (Euro)`

---

## Data Preprocessing

Several preprocessing steps were applied before training the models.

### Cleaning

* Removed unnecessary columns:

  * `id`
  * `Product`

### Categorical Encoding

Used:

* `pd.get_dummies()`

Applied to:

* Company
* TypeName
* OpSys
* CPU_Company
* GPU_Company
* Memory Type

Purpose:
Convert categorical text data into numerical format suitable for machine learning models.

---

## Feature Engineering

### Screen Resolution Processing

Extracted:

* Screen Width
* Screen Height

From:

```python
1920x1080
```

Purpose:
Allow models to understand display quality numerically.

---

### Memory Conversion

Converted:

* GB → MB
* TB → MB

Purpose:
Standardize storage units into one consistent numeric scale.

Example:

* 256GB → 256000 MB
* 1TB → 1000000 MB

---

## Exploratory Data Analysis (EDA)

Visualizations included:

### Correlation Heatmap

Used to identify:

* Strongly correlated features with laptop price
* Feature relationships

### Histograms

Analyzed:

* Price distribution

### Boxplots

Detected:

* Outliers in laptop prices

---

## Feature Selection

Used correlation analysis to select the most relevant features:

```python
target_corr = data.corr()["Price (Euro)"]
```

Top correlated features were selected for training.

Purpose:

* Reduce noise
* Improve generalization
* Reduce overfitting
* Improve training efficiency

---

## Machine Learning Models Implemented

### Linear Regression

Basic regression baseline model.

Concept:
Fits a straight linear relationship between features and price.

---

### Polynomial Regression

Used:

```python
PolynomialFeatures(degree=2)
```

Purpose:
Capture nonlinear relationships between laptop specifications and price.

Example:

* RAM × CPU interaction
* Screen size × GPU interaction

---

### Ridge Regression

Linear regression with L2 regularization.

Purpose:
Reduce overfitting by penalizing large weights.

Key parameter:

* `alpha`

Large alpha:

* More regularization
* Higher bias
* Lower variance

Small alpha:

* Less regularization
* Lower bias
* Higher variance

---

### Lasso Regression

Linear regression with L1 regularization.

Purpose:

* Reduce overfitting
* Perform automatic feature selection

Special property:
Can shrink some coefficients exactly to zero.

---

### Decision Tree Regressor

Nonlinear regression model using recursive feature splits.

Advantages:

* Captures complex patterns
* Handles nonlinear relationships

Disadvantages:

* Can easily overfit

---

## Pipelines

Used:

```python
Pipeline()
```

Purpose:
Combine preprocessing and model training into one workflow.

Example:

* Polynomial feature generation
* Scaling
* Regression model

Benefits:

* Cleaner workflow
* Prevents data leakage
* Ensures preprocessing happens inside each CV fold

---

## Feature Scaling

Used:

```python
StandardScaler()
```

Purpose:
Normalize feature distributions.

Why important:
Models like Ridge, Lasso, and Polynomial Regression are sensitive to feature scale.

---

## Log Transformation

Applied:

```python
y_log = np.log1p(y)
```

Purpose:
Reduce skewness in laptop prices.

Benefits:

* Stabilizes variance
* Reduces effect of extreme prices
* Improves regression performance

---

## Cross Validation

Used:

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

Purpose:
Evaluate model performance more reliably.

Process:

* Split data into 5 folds
* Train on 4 folds
* Validate on 1 fold
* Repeat 5 times

Advantages:

* Better generalization estimate
* Reduces overfitting risk
* Uses more data efficiently

---

## Evaluation Metrics

### R² Score

Measures how much variance the model explains.

Range:

* 1 → perfect prediction
* 0 → no predictive power

---

### MAE (Mean Absolute Error)

Average absolute prediction error.

Easy to interpret.

---

### RMSE (Root Mean Squared Error)

Punishes large errors more heavily.

Useful when large pricing mistakes are costly.

---

## Learning Curves

Used:

```python
learning_curve()
```

Purpose:
Analyze:

* Overfitting
* Underfitting
* Bias vs Variance

### Interpretation

#### Overfitting

* Low training error
* High validation error

Meaning:
Model memorizes training data.

---

#### Underfitting

* High training error
* High validation error

Meaning:
Model too simple.

---

#### Good Fit

* Training and validation curves converge closely.

Meaning:
Good generalization.

---

## Hyperparameter Tuning

Used:

```python
GridSearchCV()
```

Purpose:
Find optimal model parameters.

---

## Ridge Regression Tuning

Tuned:

```python
alpha
```

Grid:

```python
[0.01, 0.1, 1, 10, 50, 100]
```

Purpose:
Find best regularization strength.

---

## Polynomial Regression Tuning

Tuned:

```python
degree
```

Grid:

```python
[1, 2, 3, 4]
```

Purpose:
Find best polynomial complexity.

Higher degree:

* More flexible
* Higher variance
* Higher overfitting risk

Lower degree:

* Simpler model
* Higher bias

---

## Test Data Preprocessing

Created reusable preprocessing function:

```python
preprocess_data()
```

Purpose:
Ensure test data receives the exact same transformations as training data.

Critical for:

* Consistent feature structure
* Correct predictions

---

## Final Prediction Pipeline

Steps:

1. Preprocess training data
2. Train regression models
3. Evaluate using cross-validation
4. Tune hyperparameters
5. Select best model
6. Preprocess test dataset
7. Generate predictions
8. Export Kaggle submission file

---

## Libraries and Technologies Used

### Data Handling

* `pandas`
* `numpy`

### Visualization

* `matplotlib`
* `seaborn`

### Machine Learning

* `scikit-learn`

Models used:

* LinearRegression
* Ridge
* Lasso
* DecisionTreeRegressor
* PolynomialFeatures

---

## Key Machine Learning Concepts Demonstrated

* Regression
* Feature Engineering
* One-Hot Encoding
* Regularization
* Overfitting vs Underfitting
* Bias vs Variance Tradeoff
* Cross Validation
* Hyperparameter Tuning
* Learning Curves
* Pipeline Construction
* Error Metrics
* Log Transformation

---

## Example GitHub Repository Tags

`machine-learning` `regression` `laptop-price-prediction` `scikit-learn` `ridge-regression` `lasso-regression` `decision-tree` `polynomial-regression` `feature-engineering` `cross-validation` `python`

---

## Short GitHub README Summary

This project develops a complete machine learning regression pipeline for predicting laptop prices using hardware specifications and laptop features. The workflow includes data preprocessing, feature engineering, one-hot encoding, log transformation, feature selection, cross-validation, learning curves, and hyperparameter tuning. Multiple regression models were implemented and compared, including Linear Regression, Polynomial Regression, Ridge Regression, Lasso Regression, and Decision Tree Regression. Model performance was evaluated using R², MAE, and RMSE metrics to identify the most effective pricing model.
