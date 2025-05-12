```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (assuming it's in a CSV file)
def read_dataset(file_path):
    return pd.read_csv(file_path)

# Main function to execute the pipeline
def main():
    # Path to your dataset file
    file_path = 'path_to_your_dataset.csv'
    
    # Load data
    df = read_dataset(file_path)
    
    # Assuming the target variable is named 'target'
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the pipeline steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    classifier = RandomForestClassifier()
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                                ('classifier', classifier)])
    
    # Define the hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Fit the model with GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Get the best estimator
    best_model = grid_search.best_estimator_
    
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

# Call the main function
if __name__ == "__main__":
    main()
```

This code is a comprehensive implementation of a machine learning pipeline for predicting myocardial infarction complications. It includes data loading, feature engineering, model training with hyperparameter tuning, and evaluation. The pipeline uses a Random Forest Classifier and applies both numerical scaling and one-hot encoding to handle different types of features effectively.

### Key Components:
1. **Data Loading**: The dataset is loaded using a custom function `read_dataset`.
2. **Feature Engineering**: The pipeline includes steps for preprocessing, where numerical features are scaled and categorical features are one-hot encoded.
3. **Pipeline Definition**: A scikit-learn `Pipeline` object is created to encapsulate the data preprocessing and model training.
4. **Hyperparameter Tuning**: GridSearchCV is used to find the best hyperparameters for the Random Forest Classifier.
5. **Model Training and Evaluation**: The model is trained on the training data, and its performance is evaluated on the test data.

### Usage:
- Replace `'path_to_your_dataset.csv'` with the actual path to your dataset file.
- Ensure that the target variable is named 'target' or adjust the code accordingly.

This pipeline provides a robust approach to building and evaluating a machine learning model for predicting myocardial infarction complications based on given features.