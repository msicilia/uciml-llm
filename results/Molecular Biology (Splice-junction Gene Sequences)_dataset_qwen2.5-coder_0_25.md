The provided Python script demonstrates a typical workflow for building a machine learning model on a dataset. Here's a breakdown of each step in the script:

1. **Loading the Dataset**:
   - The `read_dataset` function is defined to load features and targets from CSV files located at the specified file path.
   - It reads the features (`features.csv`) and targets (`targets.csv`) into pandas DataFrames.
   - The target values are converted to a 1D array using `.values.ravel()`.

2. **Preprocessing**:
   - All categorical columns in the feature DataFrame are encoded using `LabelEncoder`. This converts each category into a numerical label.
   - The resulting encoded features are stored in the `features` variable, and the original encoders are stored in the `label_encoders` dictionary for potential future use (e.g., decoding predictions).

3. **Splitting the Data**:
   - The dataset is split into training and testing sets using `train_test_split`. The test set comprises 20% of the data (`test_size=0.2`), with a random state of 42 for reproducibility.

4. **Model Selection and Training**:
   - A `RandomForestClassifier` is instantiated with 100 trees and a maximum depth of 10.
   - The model is trained on the training data using the `.fit()` method, passing in the training features (`X_train`) and their corresponding target labels (`y_train`).

5. **Evaluation**:
   - After training, the model's performance is evaluated on the test set using `accuracy_score` and `f1_score`.
   - The accuracy score measures the proportion of correct predictions.
   - The F1 score provides a balance between precision and recall for multi-class classification problems.

6. **Model Persistence**:
   - The trained model and label encoders are saved to disk using `joblib.dump`. This allows the model and its associated encoding information to be easily loaded later without needing to retrain from scratch.

### Example Output
When you run this script, it will output something like:
```
Accuracy: 0.9524
F1 Score: 0.9527
```
These values indicate the model's performance on the test set in terms of accuracy and F1 score.

### Note
- Ensure that the CSV files (`features.csv` and `targets.csv`) are located at the specified path (`"./datasets/Molecular Biology (Splice-junction Gene Sequences)/"`) or update the file path accordingly.
- The script assumes that all features are categorical. If your data contains numerical features, you might not need encoding, but this should be verified based on the actual dataset.

This script provides a solid foundation for building and evaluating a machine learning model using Python's pandas and scikit-learn libraries.