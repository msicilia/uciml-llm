```python
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load dataset
features, targets = read_dataset("./datasets/Statlog (Shuttle)/")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

# Preprocess the data by scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define the classifier and hyperparameter distribution for RandomizedSearchCV
svm_classifier = SVC(random_state=42)
param_dist = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Perform Random Search to find the best hyperparameters
random_search = RandomizedSearchCV(svm_classifier, param_dist, n_iter=10, cv=5, random_state=42)
random_search.fit(X_train_scaled, y_train)

# Get the best model and evaluate its performance on the validation set
best_model = random_search.best_estimator_
val_accuracy = best_model.score(X_val_scaled, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")
```