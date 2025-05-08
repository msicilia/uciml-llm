To improve the performance of our classification models and potentially achieve a higher accuracy score, we can perform hyperparameter tuning. Hyperparameter tuning involves systematically searching through a manually specified subset of the hyperparameter space to find the optimal parameters for a particular estimator.

We will use GridSearchCV from scikit-learn to perform grid search on the RandomForestClassifier. GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the `param_grid` parameter and evaluates each candidate using cross-validation.

Here’s how you can implement it:

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Random Forest Classifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Create a GridSearchCV object with the parameter grid and scoring method
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_best_rf = best_rf_model.predict(X_test)
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)

print(f"Best Parameters: {best_params}")
print(f"Best Random Forest Accuracy: {accuracy_best_rf}")
```

### Explanation:

1. **Define Parameter Grid**: We define a dictionary `param_grid` that specifies the hyperparameters to tune and their possible values.
2. **Initialize RandomForestClassifier**: We initialize an instance of `RandomForestClassifier`.
3. **GridSearchCV**: We create a `GridSearchCV` object by passing in the classifier, parameter grid, cross-validation method (`cv=3` for 3-fold cross-validation), scoring method (`scoring='accuracy'`), and number of jobs to run in parallel (`n_jobs=-1`).
4. **Fit GridSearchCV**: We fit the `GridSearchCV` object to our training data using `fit(X_train, y_train)`.
5. **Best Parameters and Best Estimator**: After fitting, we retrieve the best parameters found by grid search and the corresponding best estimator.
6. **Evaluate Best Model**: Finally, we evaluate the performance of the best model on the test set.

### Notes:
- GridSearchCV can be computationally expensive, especially if the parameter grid is large or the dataset is large. In such cases, consider using RandomizedSearchCV, which samples a fixed number of candidates from a parameter space with specified distributions.
- The `cv` parameter in GridSearchCV controls the number of folds for cross-validation. A larger value can help reduce variance but may increase computation time.

By performing hyperparameter tuning, we should be able to find a combination of parameters that results in higher accuracy on the test set.