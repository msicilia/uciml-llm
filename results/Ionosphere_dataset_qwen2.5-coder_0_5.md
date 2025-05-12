It appears that the response provided includes nested calls to execute Python code, which is not necessary and might lead to confusion. Let me simplify this for you:

1. **Loading Dataset**: Ensure you have the correct dataset loaded.
2. **Splitting Data**: Use `train_test_split` to divide your data into training and testing sets.

Here's how you can do it:

```python
# Assuming read_dataset is a function that loads your dataset
features, targets = read_dataset("./datasets/Ionosphere/")

# Convert targets to 1D array if necessary (depending on the library used)
targets = targets.values.flatten()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
```

3. **Model Initialization**: Create instances of your models.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Initialize models with different hyperparameters
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'SVC': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3, weights='distance')
}
```

4. **Training and Evaluation**: Loop through the models to train and evaluate each one.

```python
from sklearn.metrics import accuracy_score, classification_report

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_rep)
```

This should give you a clear comparison of the performance of each model on your dataset. Make sure that `read_dataset` is correctly implemented and accessible from where this code is executed.