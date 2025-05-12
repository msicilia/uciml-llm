Here is a detailed breakdown of the script provided:

1. **Importing Necessary Libraries**:
   - The script starts by importing essential libraries such as `pandas`, `train_test_split` from `sklearn.model_selection`, `StandardScaler` and `OneHotEncoder` from `sklearn.preprocessing`, `ColumnTransformer` and `Pipeline` from `sklearn.compose`, `SimpleImputer` from `sklearn.impute`, and `accuracy_score` and `classification_report` from `sklearn.metrics`.

2. **Loading Data**:
   - The dataset is loaded using a custom function `read_dataset` with the path `"./datasets/National Poll on Healthy Aging (NPHA)/"`.

3. **Preprocessing Data**:
   - The target variable `targets` is converted to a 1D array.
   - Numerical and categorical features are identified, and transformers are created for each type of feature.
     - For numerical features, a pipeline with imputation using the mean value and scaling is defined.
     - For categorical features, a pipeline with imputation using the most frequent value and one-hot encoding is defined.
   - A `ColumnTransformer` combines these transformers into a single preprocessor.

4. **Splitting Data**:
   - The dataset is split into training and testing sets with an 80/20 ratio using `train_test_split`.

5. **Building Models**:
   - A logistic regression model (`log_reg`) and a random forest classifier (`rf_classifier`) are created, each wrapped in a pipeline that includes the preprocessor.

6. **Training Models**:
   - Both models are trained on the training data using the `.fit()` method.

7. **Evaluating Models**:
   - Predictions are made on the test set for both models.
   - The accuracy and classification report for each model are printed.

8. **Hyperparameter Tuning**:
   - A parameter grid is defined for the random forest classifier to explore different combinations of `n_estimators`, `max_depth`, and `min_samples_split`.
   - A `GridSearchCV` object is created with the random forest classifier, parameter grid, and 5-fold cross-validation.
   - The model with the best parameters is trained on the entire training data using `.fit()`.

9. **Final Evaluation**:
   - The final model with the best parameters is evaluated on the test set.
   - The accuracy and classification report for the final best model are printed.

### Summary
This script provides a comprehensive workflow for building, evaluating, and tuning machine learning models. It ensures that each step of the process is well-documented and reproducible, making it easier to understand and modify as needed.