```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Perform Grid Search
grid_search = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate on test set with best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
print("Classification Report (Best Model):")
print(classification_report(y_test, y_pred_best))
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Accuracy (Best Model): {accuracy_best:.4f}")
```