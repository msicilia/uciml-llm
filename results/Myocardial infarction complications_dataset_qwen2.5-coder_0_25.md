Certainly! Below is the complete code to perform hyperparameter tuning using `GridSearchCV` for a RandomForestClassifier on your dataset. This script includes reading the dataset, preprocessing steps, and model evaluation.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
features, targets = read_dataset("./datasets/Myocardial infarction complications/")  # Ensure this function is defined correctly

# Convert target to 1D array
targets = targets.values.ravel()

# Separate features and target
X = features
y = targets

# Split the data into training and testing sets (80-20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Define preprocessing steps for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [col for col in X.columns if col not in categorical_cols]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create a pipeline that includes preprocessing and the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best estimator and its score
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_

print(f"Best Model: {best_model}")
print(f"Best Score: {best_score:.4f}")

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_rep)
```

### Explanation:
1. **Reading the Dataset**: Ensure that `read_dataset` is a function or method you have defined to load your dataset.
2. **Preprocessing**:
   - Categorical columns are identified and handled using `OneHotEncoder`.
   - Numerical columns are scaled using `StandardScaler`.
3. **Model Creation**: A pipeline is created to combine the preprocessing steps with the RandomForestClassifier.
4. **Hyperparameter Tuning**: `GridSearchCV` is used to search for the best hyperparameters within the specified grid.
5. **Evaluation**: The best model is evaluated on the test set, and performance metrics (accuracy and classification report) are printed.

Make sure that your dataset file path and any other specific details are correctly configured before running the script.