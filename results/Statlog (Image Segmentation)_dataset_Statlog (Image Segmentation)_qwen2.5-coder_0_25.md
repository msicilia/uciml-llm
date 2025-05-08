It seems like you are trying to evaluate multiple machine learning models on a dataset and determine which one performs the best. Here's how you can do it using Python:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Read the dataset
features, targets = read_dataset("./datasets/Statlog (Image Segmentation)/")
targets = targets.values.ravel()  # Convert to 1D array

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, targets, test_size=0.2, random_state=42)

# Define models to evaluate
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC()
}

# Evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    results[name] = {'Accuracy': accuracy, 'F1-Score': f1}

# Print results
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  F1-Score: {metrics['F1-Score']:.4f}")

# Choose the best model based on accuracy or F1-score
best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
best_model = models[best_model_name]

print(f"\nThe best model is {best_model_name} with Accuracy: {results[best_model_name]['Accuracy']:.4f}")
```

### Explanation:

1. **Reading the Dataset**: The `read_dataset` function reads the dataset and returns features and targets.
2. **Feature Scaling**: The features are scaled using `StandardScaler` to ensure that each feature contributes equally to the distance calculations in algorithms like SVM.
3. **Train-Test Split**: The data is split into training and testing sets with an 80-20 ratio.
4. **Defining Models**: A dictionary of models is defined, including Logistic Regression, Random Forest, and SVM.
5. **Evaluating Models**: Each model is trained on the training data and evaluated on the test data using accuracy and F1-score.
6. **Printing Results**: The performance of each model is printed.
7. **Selecting the Best Model**: The best model is selected based on the highest accuracy.

### Notes:
- Ensure that `read_dataset` function is correctly implemented to read your dataset.
- You can add more models to the dictionary if needed.
- Accuracy and F1-score are commonly used metrics for classification problems, but you might want to consider other metrics depending on the problem requirements.