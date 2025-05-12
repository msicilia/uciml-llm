It looks like there was an interruption in the response. Let me provide you with a more detailed and clear version of the script:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load and Prepare Data
# Replace 'read_dataset' with the actual function or method you use to load your dataset
features, targets = read_dataset("./datasets/Glass Identification/")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features.drop('Id_number', axis=1), 
                                                    targets['Type_of_glass'], 
                                                    test_size=0.2, 
                                                    random_state=42)

# Step 2: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Model Selection and Training
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train)

# Step 4: Evaluate Model
y_pred = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

### Explanation of Each Step:

1. **Loading and Preparing Data**:
   - The `read_dataset` function is used to load your dataset.
   - The dataset is split into features (`X`) and target labels (`y`).
   - The data is further split into training and testing sets using `train_test_split`.

2. **Feature Scaling**:
   - Features are scaled using `StandardScaler` to ensure that each feature contributes equally to the distance computations in the Random Forest algorithm.

3. **Model Selection and Training**:
   - A `RandomForestClassifier` is instantiated with 100 trees (`n_estimators=100`) and trained on the scaled training data.

4. **Evaluating the Model**:
   - The model is evaluated on the test set using accuracy, classification report (which includes precision, recall, f1-score, and support for each class), and the confusion matrix.

### Running the Script:

To run this script, ensure you have the necessary libraries installed:

```sh
pip install pandas scikit-learn
```

Replace `read_dataset("./datasets/Glass Identification/")` with the actual function or method you use to load your dataset. This might involve reading a CSV file, loading data from a database, etc.

### Example of Loading Data from a CSV File:

If your dataset is stored in a CSV file named `Glass_Identification.csv`, you can use pandas to load it:

```python
import pandas as pd

# Load the dataset from a CSV file
features = pd.read_csv("./datasets/Glass_Identification.csv", drop="Id_number")
targets = pd.read_csv("./datasets/Glass_Identification.csv", usecols=["Type_of_glass"])
```

Make sure to adjust the column names and paths according to your actual dataset.

This script should provide you with a detailed evaluation of your Random Forest model's performance on the test set.