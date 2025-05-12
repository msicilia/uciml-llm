```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset using the appropriate tool
features, targets = read_dataset("./datasets/Molecular Biology (Splice-junction Gene Sequences)/")

# Convert target to 1d array if necessary
targets = targets.values.ravel()

# Encode the labels as integers
label_encoder = LabelEncoder()
encoded_targets = label_encoder.fit_transform(targets)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, encoded_targets, test_size=0.2, random_state=42)

# Initialize a classification model (e.g., RandomForestClassifier)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```