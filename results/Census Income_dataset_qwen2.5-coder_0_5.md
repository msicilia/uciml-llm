To execute the provided Python code for preprocessing, feature selection, and model evaluation, you can use a standard Python interpreter or an interactive Python environment like Jupyter Notebook. Here's how you can run it:

1. **Using a Standard Python Interpreter**:
   - Save the code in a file with a `.py` extension, for example, `census_income_analysis.py`.
   - Open a terminal or command prompt.
   - Navigate to the directory where the file is saved.
   - Run the script using the following command:
     ```sh
     python census_income_analysis.py
     ```

2. **Using Jupyter Notebook**:
   - Install Jupyter Notebook if you haven't already:
     ```sh
     pip install notebook
     ```
   - Open a terminal or command prompt.
   - Navigate to the directory where the file is saved.
   - Start Jupyter Notebook:
     ```sh
     jupyter notebook
     ```
   - This will open a web browser with the Jupyter Notebook interface. Create a new Python notebook and paste the code into a cell, then run it.

Here's the complete code for reference:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Read the dataset
features, targets = read_dataset("./datasets/Census Income/")

# Separate categorical and numerical columns
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Define preprocessing steps
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Apply preprocessing to the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

from sklearn.feature_selection import SelectKBest, chi2

# Apply feature selection
kbest = SelectKBest(score_func=chi2, k='all')
X_train_selected = kbest.fit_transform(X_train_processed, y_train)
X_test_selected = kbest.transform(X_test_processed)

selected_features = features.columns[kbest.get_support()]
print("Selected Features:", selected_features)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define models and parameter grids
models = {
    'RandomForest': (RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
    'GradientBoosting': (GradientBoostingClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
    'SVM': (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})
}

# Train and evaluate models
for name, (model, param_grid) in models.items():
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_selected, y_train)
    print(f'{name}: Best Parameters: {grid_search.best_params_}, Best Score: {grid_search.best_score_}')

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_selected)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
```

Make sure you have the necessary libraries installed and that the dataset is correctly formatted. If you encounter any issues, please provide the error messages so I can assist further.