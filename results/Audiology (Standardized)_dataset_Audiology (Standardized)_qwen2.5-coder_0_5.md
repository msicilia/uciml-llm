```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to read dataset
def read_dataset(file_path):
    return pd.read_csv(file_path)

# Function to split data into training and testing sets
def split_data(data, test_size=0.2):
    X = data.drop('target', axis=1)
    y = data['target']
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Function to perform grid search for hyperparameter tuning
def grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_estimator_

# Function to train the model using best hyperparameters
def train_model(X_train, y_train, best_params):
    rf = RandomForestClassifier(**best_params, random_state=42)
    rf.fit(X_train, y_train)
    return rf

# Function to evaluate the model on test data
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

# Main execution flow
if __name__ == "__main__":
    # Step 1: Read the dataset
    file_path = './datasets/Audiology (Standardized)/data.csv'
    data = read_dataset(file_path)

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data)

    # Step 3: Perform grid search for hyperparameter tuning
    best_params, best_model = grid_search(X_train, y_train)
    print("Best Hyperparameters:", best_params)

    # Step 4: Train the model using the best hyperparameters
    trained_model = train_model(X_train, y_train, best_params)

    # Step 5: Evaluate the model on test data
    accuracy, report = evaluate_model(trained_model, X_test, y_test)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
```