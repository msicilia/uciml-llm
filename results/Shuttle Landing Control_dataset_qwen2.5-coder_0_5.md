To build a classification model for the given dataset using scikit-learn, follow these steps:

1. **Load the dataset**:
   - Use the `read_dataset` function to load the features and targets.

2. **Handle missing values**:
   - Fill missing values in the dataset with the mean of each respective column.

3. **Convert target variable to a 1D array**:
   - Ensure that the target variable is converted to a 1D array for compatibility with scikit-learn functions.

4. **Split the data into training and testing sets**:
   - Use `train_test_split` from `sklearn.model_selection` to split the dataset into training and testing sets.

5. **Choose an appropriate classification algorithm**:
   - For a tiny dataset, Gaussian Naive Bayes (GaussianNB) is a good starting point due to its simplicity and assumptions about the data distribution.

6. **Train the model on the training set**:
   - Fit the chosen classifier to the training data.

7. **Evaluate the model on the testing set**:
   - Use metrics like accuracy to evaluate the performance of the model.

Here is the complete code:

```python
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
features, targets = read_dataset("./datasets/Shuttle Landing Control/")

# Handle missing values by filling them with the mean of each respective column
features.fillna(features.mean(), inplace=True)

# Convert target variable to a 1D array
targets = targets.values.ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Choose an appropriate classification algorithm (Naive Bayes in this case)
model = GaussianNB()

# Train the model on the training set
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
```

### Explanation:
- **Loading the Dataset**: The `read_dataset` function loads the features and targets from a specified path.
- **Handling Missing Values**: We fill missing values in the dataset using the mean of each column to ensure that no data is lost during training.
- **Converting Target Variable**: The target variable is converted to a 1D array for compatibility with scikit-learn functions.
- **Splitting Data**: We split the data into training and testing sets using `train_test_split`, where 20% of the data is used for testing.
- **Model Selection**: Gaussian Naive Bayes (`GaussianNB`) is chosen as it is suitable for datasets with continuous features and assumes that the features are normally distributed.
- **Training and Evaluation**: The model is trained on the training data and then evaluated on the testing data using accuracy as the metric.

This approach provides a basic framework to build and evaluate a classification model using scikit-learn. Depending on the dataset characteristics, you might want to explore other algorithms or preprocessing steps for better performance.