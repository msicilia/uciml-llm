The provided code snippet is a Python script that loads a dataset, encodes categorical variables, splits the data into training and testing sets, trains two machine learning models (RandomForest and GradientBoosting), evaluates their performance using accuracy as the metric, and prints out the results. The script uses popular libraries such as pandas, sklearn.model_selection, sklearn.preprocessing, sklearn.ensemble, and sklearn.metrics.

Here's a breakdown of the key components:

1. **Loading the Dataset**:
   ```python
   features, targets = read_dataset("./datasets/Balloons/")
   ```
   This line assumes that there is a function `read_dataset` that reads the dataset from a specified path and returns the features (inputs) and targets (outputs).

2. **Encoding Categorical Variables**:
   ```python
   le = LabelEncoder()
   for column in features.columns:
       if features[column].dtype == type(object):
           features[column] = le.fit_transform(features[column])
   ```
   This loop iterates over each column in the feature dataset and encodes categorical variables using `LabelEncoder`.

3. **Splitting the Data**:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
   ```
   This line splits the dataset into training and testing sets with 80% of the data used for training and 20% for testing.

4. **Training Models**:
   ```python
   models = {
       "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
       "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
   }
   for name, model in models.items():
       model.fit(X_train, y_train)
   ```
   This part creates a dictionary of two models: RandomForest and GradientBoosting. It then trains each model on the training data.

5. **Evaluating Models**:
   ```python
   results = {}
   for name, model in models.items():
       y_pred = model.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)
       results[name] = accuracy
   ```
   This loop predicts the target values using each trained model on the test data and calculates the accuracy of each model.

6. **Printing Results**:
   ```python
   print("Model Accuracy:")
   for name, accuracy in results.items():
       print(f"{name}: {accuracy:.4f}")
   ```
   Finally, the script prints out the accuracy of each model.

### Summary
The script effectively demonstrates how to load and preprocess data, split it into training and testing sets, train machine learning models, evaluate their performance, and output the results. This is a common workflow in machine learning projects.