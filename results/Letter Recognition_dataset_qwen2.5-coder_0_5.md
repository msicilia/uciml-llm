It seems there was a misunderstanding in the previous response. The code snippet you provided is indeed correct and will perform a grid search to find the optimal hyperparameters for an SVM classifier using cross-validation. However, the `X_train`, `y_train`, `X_test` variables should be defined before running this script.

Here's a complete example that includes loading the dataset, splitting it into training and testing sets, applying feature scaling, and then performing the grid search:

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load a sample dataset (replace with your own dataset)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVM classifier
svm_classifier = SVC()

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters found and the best score
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate the best model on the test data
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test_scaled, y_test)
print("Test accuracy with best model:", test_score)
```

### Explanation:
1. **Loading Data**: The script uses the Iris dataset from `sklearn.datasets`. You should replace this with your own dataset.
2. **Data Splitting**: The dataset is split into training and testing sets using `train_test_split`.
3. **Feature Scaling**: Feature scaling is applied to ensure that all features have a similar scale, which is crucial for SVMs.
4. **Grid Search**: A grid search is performed using `GridSearchCV` to find the optimal hyperparameters (`C`, `kernel`, and `gamma`) for the SVM classifier.
5. **Evaluation**: The best model found by the grid search is evaluated on the test set.

Make sure you have the necessary libraries installed (`sklearn`, `numpy`, etc.) before running this script. You can install them using pip if needed:

```sh
pip install scikit-learn numpy pandas matplotlib
```

This should give you a complete solution for optimizing an SVM classifier using grid search and cross-validation.