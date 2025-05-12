It looks like you have provided a code snippet for training and evaluating a Random Forest Classifier using GridSearchCV. However, there are a few issues in the code that need to be addressed:

1. **LabelEncoder Import**: The `LabelEncoder` class is part of `sklearn.preprocessing`, so you should import it from there.
2. **Function Definition**: The `read_dataset` function should return both features and targets.

Here's the corrected version of your code:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Placeholder for the read_dataset function
def read_dataset(file_path):
    # Replace this with your actual implementation to load and return the dataset
    df = pd.read_csv(file_path)
    X = df.drop('target_column', axis=1)  # Assuming 'target_column' is the target variable
    y = df['target_column']
    return X, y

# Load the dataset
features, targets = read_dataset("./datasets/Horse_Colic.csv")

# Handle missing values and feature scaling
continuous_cols = [col for col in features.columns if features[col].dtype == float]
discrete_nominal_cols = [col for col in features.columns if features[col].dtype == object]

features[continuous_cols] = features[continuous_cols].fillna(features[continuous_cols].mean())
le = LabelEncoder()
for col in discrete_nominal_cols:
    features[col] = le.fit_transform(features[col])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform GridSearchCV
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_rf_classifier = grid_search.best_estimator_

# Evaluate the model on test set
y_pred = best_rf_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
```

### Explanation of Changes:
1. **Import LabelEncoder**: Added `from sklearn.preprocessing import LabelEncoder`.
2. **Function Definition**: The `read_dataset` function now returns both features and targets.
3. **Target Column**: Assuming `'target_column'` is the name of your target variable, replace it with the actual column name from your dataset.

Make sure to replace `"./datasets/Horse_Colic.csv"` with the correct file path to your dataset. Also, adjust the `target_column` placeholder with the actual target variable in your dataset.