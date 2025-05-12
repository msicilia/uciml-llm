```
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Read the dataset
features, targets = read_dataset("./datasets/Tennis Major Tournament Match Statistics/")

# Convert target to a 1D array if it's not already
targets = targets.values.ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=200)
param_grid_log_reg = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5)
grid_search_log_reg.fit(X_train_scaled, y_train)

best_log_reg = grid_search_log_reg.best_estimator_
y_pred_log_reg = best_log_reg.predict(X_test_scaled)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Accuracy: {accuracy_log_reg:.4f}")

# Model 2: Random Forest
rf_clf = RandomForestClassifier()
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(rf_clf, param_grid_rf, cv=5)
grid_search_rf.fit(X_train_scaled, y_train)

best_rf_clf = grid_search_rf.best_estimator_
y_pred_rf = best_rf_clf.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# Model 3: Support Vector Machine
svc = SVC(probability=True)
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5)
grid_search_svc.fit(X_train_scaled, y_train)

best_svc = grid_search_svc.best_estimator_
y_pred_svc = best_svc.predict(X_test_scaled)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f"SVM Accuracy: {accuracy_svc:.4f}")

# Best Model
best_model_accuracy = max(accuracy_log_reg, accuracy_rf, accuracy_svc)
if best_model_accuracy == accuracy_log_reg:
    print("Best Model: Logistic Regression")
elif best_model_accuracy == accuracy_rf:
    print("Best Model: Random Forest")
else:
    print("Best Model: Support Vector Machine")

# Final Accuracy
final_accuracy = best_model_accuracy
print(f"Final Accuracy: {final_accuracy:.4f}")
```