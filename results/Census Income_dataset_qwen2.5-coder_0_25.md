The provided code snippet is a comprehensive example of how to build, train, evaluate, and optimize a logistic regression model using the `scikit-learn` library in Python. Here's a breakdown of each part of the script:

### 1. Import Necessary Libraries
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```
This section imports the required modules from `scikit-learn` for data preprocessing, model building, and evaluation.

### 2. Define Feature Sets
```python
numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
```
These lines define the numerical and categorical features in your dataset.

### 3. Create Preprocessor
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
```
This creates a `ColumnTransformer` that standardizes the numerical features and applies one-hot encoding to the categorical features.

### 4. Split Data
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
This splits the dataset into training and testing sets with a 80/20 split.

### 5. Create Pipeline
```python
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=200))
])
```
This creates a pipeline that first applies the preprocessor and then fits the logistic regression model.

### 6. Train Model
```python
log_reg_pipeline.fit(X_train, y_train)
```
This trains the model on the training data.

### 7. Make Predictions
```python
y_pred = log_reg_pipeline.predict(X_test)
```
This makes predictions on the test data.

### 8. Evaluate Performance (Default Model)
```python
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
```
This calculates and prints the performance metrics for the default model.

### 9. Optional: Hyperparameter Tuning
```python
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(log_reg_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

print(f"Best Model Accuracy: {accuracy_best:.2f}")
print(f"Best Model Precision: {precision_best:.2f}")
print(f"Best Model Recall: {recall_best:.2f}")
print(f"Best Model F1-score: {f1_best:.2f}")
```
This performs hyperparameter tuning using `GridSearchCV` to find the best model parameters and then evaluates the performance of the best model.

### Running the Script
To run the script, ensure that the data (`X` and `y`) is correctly loaded into your environment. You can execute the script in a Python environment as shown above.

### Notes
- Ensure that the data (`X` and `y`) is correctly loaded into your environment before running the script.
- The hyperparameter tuning step is optional but recommended for improving model performance.

This script provides a robust framework for building, training, evaluating, and optimizing machine learning models using scikit-learn.