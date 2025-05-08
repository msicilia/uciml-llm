```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred_rf = rf_classifier.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print(f"Accuracy with Random Forest: {accuracy_rf:.2f}")
print("Classification Report with Random Forest:")
print(report_rf)

# Perform feature selection
selector = SelectKBest(chi2, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Train and evaluate the model with selected features
rf_classifier_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_selected.fit(X_train_selected, y_train)

y_pred_rf_selected = rf_classifier_selected.predict(X_test_selected)
accuracy_rf_selected = accuracy_score(y_test, y_pred_rf_selected)
report_rf_selected = classification_report(y_test, y_pred_rf_selected)

print(f"Accuracy with Selected Features: {accuracy_rf_selected:.2f}")
print("Classification Report with Selected Features:")
print(report_rf_selected)

# Initialize and train the Logistic Regression Classifier with L1 regularization
log_reg_classifier = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
log_reg_classifier.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred_log_reg = log_reg_classifier.predict(X_test_scaled)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
report_log_reg = classification_report(y_test, y_pred_log_reg)

print(f"Accuracy with Logistic Regression (L1): {accuracy_log_reg:.2f}")
print("Classification Report with Logistic Regression (L1):")
print(report_log_reg)
```