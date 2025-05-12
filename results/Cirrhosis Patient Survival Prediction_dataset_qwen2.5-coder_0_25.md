```python
# Read the dataset
features, targets = read_dataset("./datasets/Cirrhosis Patient Survival Prediction/")

# Separate features and target variable
X = features.drop(columns=["ID", "Status"])
y = features["Status"].map({"C": 0, "CL": 1, "D": 2})

# Handle missing values using imputation
numerical_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(strategy="most_frequent")

X_numerical = X.select_dtypes(include=["int64", "float64"])
X_categorical = X.select_dtypes(include=["object"])

X_numerical_imputed = numerical_imputer.fit_transform(X_numerical)
X_categorical_imputed = categorical_imputer.fit_transform(X_categorical)

# One-hot encode categorical features
one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
X_categorical_encoded = one_hot_encoder.fit_transform(X_categorical)

# Combine encoded and imputed numerical features
X_combined = np.hstack([X_numerical_imputed, X_categorical_encoded])

# Standardize features to have mean=0 and std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    # Print results
    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
```