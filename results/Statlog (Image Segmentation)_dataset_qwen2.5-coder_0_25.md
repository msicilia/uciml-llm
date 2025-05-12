It seems there was an issue with the previous response. Let's provide a more complete and detailed example, including the actual dataset loading part.

Here's a full script that includes reading a dataset from a file (assuming it's in CSV format), preprocessing the data, training a RandomForestClassifier, and evaluating its performance:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to read dataset (this would be replaced with actual dataset loading logic)
def read_dataset(file_path):
    return pd.read_csv(file_path)

# Step 1: Load and Prepare the Data
file_path = "./datasets/Statlog (Image Segmentation)/data.csv"  # Replace with your actual file path
df = read_dataset(file_path)

# Assuming 'target' is the column you want to predict, and other columns are features
features = df.drop(columns=['target'])
targets = df['target']

X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Step 3: Model Selection
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 4: Evaluation
y_pred = model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

### Explanation:

1. **Reading the Dataset**: The `read_dataset` function is assumed to load data from a CSV file. You can replace this with any other method of loading your dataset.

2. **Data Splitting and Feature Scaling**:
   - We split the data into training and validation sets using `train_test_split`.
   - We then scale the features using `StandardScaler` to ensure that all features are on a similar scale.

3. **Model Training**:
   - We create an instance of `RandomForestClassifier` and train it on the scaled training data.

4. **Evaluation**:
   - We evaluate the model on the validation set using accuracy, precision, recall, and F1-score.

### Notes:

- Ensure that your dataset is correctly formatted and contains a column named 'target' (or adjust the code to match your actual target column name).
- Adjust the file path and other parameters as needed for your specific use case.
- This script assumes that all columns except 'target' are features. If your dataset has categorical variables, you may need additional preprocessing steps.

This should provide a more complete and functional example of loading a dataset, preprocessing it, training a RandomForestClassifier, and evaluating its performance.