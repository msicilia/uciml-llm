```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Read the dataset
features, targets = read_dataset("./datasets/Cirrhosis Patient Survival Prediction/")

# Convert target to 1d array
targets = targets.values.ravel()

# Handle missing values by filling with the median or mode
for column in features.columns:
    if features[column].dtype == 'object':
        features[column].fillna(features[column].mode()[0], inplace=True)
    else:
        features[column].fillna(features[column].median(), inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    label_encoders[column] = le

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Normalize continuous features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances = pd.Series(rf_classifier.feature_importances_, index=features.columns).sort_values(ascending=False)
print(feature_importances)

# Select top features based on feature importances (e.g., top 10)
selected_features = feature_importances.index[:10]
X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

# Initialize and train the Random Forest classifier with selected features
rf_classifier_selected = RandomForestClassifier(random_state=42)
rf_classifier_selected.fit(X_train_selected, y_train)

# Make predictions
y_pred = rf_classifier_selected.predict(X_test_selected)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)

# Fit GridSearchCV
grid_search.fit(X_train_selected, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)

# Make predictions using the best model
y_pred_best = best_model.predict(X_test_selected)

# Evaluate the model
accuracy_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best, average='weighted')

print("Accuracy with Best Model:", accuracy_best)
print("F1 Score with Best Model:", f1_best)
print("Classification Report with Best Model:\n", classification_report(y_test, y_pred_best))
```