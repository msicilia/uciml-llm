It seems like the code you provided is attempting to load a dataset, preprocess it, and then train several machine learning models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting). However, there are a few issues in the code that need to be addressed:

1. **Dataset Loading**: The `read_dataset` function is not defined. You need to define this function or use an existing method to load your dataset.

2. **Function Definitions**: If you want to define functions within the provided code snippet, ensure they are properly indented and defined before they are called.

3. **Evaluation Metrics**: The evaluation metrics (accuracy, precision, recall, F1-score) are printed inside a loop for each model. This means that the best model selection will be based on the last model's performance, not the overall best.

Here is a corrected version of your code with some improvements:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the dataset
features, targets = read_dataset("./datasets/Cylinder Bands/")

# Separate features and target variable
X = features.drop(columns=['band type'])
y = targets['band type'].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical and categorical features
numeric_features = ['viscosity', 'caliper', 'ink temperature', 'humidity', 'roughness', 'blade pressure', 'press speed']
categorical_features = [col for col in X.columns if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the models and their pipelines
models = {
    'Logistic Regression': Pipeline(steps=[('preprocessor', preprocessor), ('model', LogisticRegression())]),
    'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor), ('model', DecisionTreeClassifier())]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestClassifier())]),
    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor), ('model', GradientBoostingClassifier())])
}

# Train and evaluate each model
best_model = None
best_score = -1

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f'{name}:')
    print(f'  Accuracy: {accuracy:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall: {recall:.4f}')
    print(f'  F1-Score: {f1:.4f}')
    print()
    
    # Update the best model based on accuracy
    if accuracy > best_score:
        best_model = name
        best_score = accuracy

# Print the best model and its score
print(f'\nBest Model: {best_model} with Accuracy: {best_score:.4f}')
```

### Key Improvements:
1. **Dataset Loading**: Ensure that `read_dataset` is defined or use a standard method to load your dataset.
2. **Best Model Selection**: The best model is now selected based on the highest accuracy across all models.
3. **Code Formatting**: Improved code formatting for better readability.

Make sure you have the necessary libraries installed (`pandas`, `scikit-learn`) and that your dataset is correctly formatted and accessible at the specified path.