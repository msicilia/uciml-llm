To evaluate the performance of a machine learning model on your dataset using Python's `sklearn` library, follow these steps:

1. **Load the Dataset**: Use a custom function or load data from a file.
2. **Split the Data**: Divide the dataset into training and testing sets.
3. **Initialize the Model**: Set up a Logistic Regression model.
4. **Train the Model**: Fit the model on the training data.
5. **Make Predictions**: Use the trained model to make predictions on the test set.
6. **Evaluate Performance**: Calculate accuracy and F1-score to assess the model's performance.

Here is the complete code:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load the dataset (assuming read_dataset is a custom function that loads the data)
features, targets = read_dataset("./datasets/Ionosphere/")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy and F1-score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"F1-Score: {f1:.2f}")
```

### Explanation:
1. **Loading the Dataset**: Replace `read_dataset("./datasets/Ionosphere/")` with your actual data loading function or directly load data from a file.
2. **Splitting the Data**: The dataset is split into training and testing sets using 80% of the data for training and 20% for testing. You can adjust the `test_size` parameter as needed.
3. **Initializing the Model**: A Logistic Regression model is created with a maximum of 200 iterations to ensure convergence.
4. **Training the Model**: The model is trained on the training data using the `fit` method.
5. **Making Predictions**: The trained model makes predictions on the test set using the `predict` method.
6. **Evaluating Performance**: Accuracy and F1-score are calculated to evaluate how well the model performs.

Make sure you have the necessary libraries installed. You can install them using pip if you haven't already:

```sh
pip install scikit-learn pandas numpy
```

This script provides a straightforward way to assess the performance of a logistic regression model on your dataset.