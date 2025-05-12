```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Assuming read_dataset is a function that reads data from a file and returns features and targets
features, targets = read_dataset("./datasets/Mushroom/")
X = features.drop('poisonous', axis=1)
y = features['poisonous']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifying categorical columns for preprocessing
categorical_features = X.select_dtypes(include=['object']).columns

# Defining the preprocessing steps including OneHotEncoder for categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Building the pipeline that includes preprocessing and logistic regression
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(solver='liblinear'))])

# Training the pipeline on the training data
pipeline.fit(X_train, y_train)

# Evaluating the trained model on the test data
score = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {score}")

# Defining the parameter grid for hyperparameter tuning
param_grid = {
    'classifier__C': [0.1, 1, 10],  
    'classifier__penalty': ['l1', 'l2']
}

# Performing Grid Search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Getting the best estimator from grid search
best_model = grid_search.best_estimator_

# Printing the best parameters found
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluating the best model on the test data
best_score = best_model.score(X_test, y_test)
print(f"Test Accuracy with Best Model: {best_score}")
```