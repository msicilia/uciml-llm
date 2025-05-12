The provided code performs a comprehensive hyperparameter tuning using `GridSearchCV` for the `GradientBoostingClassifier`. This approach helps in finding the optimal set of parameters that maximize the model's accuracy on the validation set. Below is a step-by-step explanation and the expected output:

1. **Data Splitting**: The dataset is split into training and validation sets with an 80-20 ratio.
   
2. **Data Preprocessing**: Both the training and validation sets are scaled using `StandardScaler` to ensure that all features contribute equally to the model.

3. **Parameter Grid Definition**: A parameter grid is defined containing different values for `n_estimators`, `learning_rate`, and `max_depth`. These parameters control the complexity and performance of the gradient boosting model.

4. **GridSearchCV Initialization**: The `GradientBoostingClassifier` is initialized, and `GridSearchCV` is set up with a 5-fold cross-validation scheme to evaluate each combination of hyperparameters.

5. **Model Training**: `GridSearchCV` iteratively trains the model for each combination of parameters in the grid and evaluates its performance on the validation set using accuracy as the scoring metric.

6. **Best Parameters Selection**: The best combination of hyperparameters is selected based on the highest cross-validated accuracy.

7. **Performance Evaluation**: The best model is evaluated on the validation set, and its accuracy is printed along with the best parameters found during the grid search.

### Expected Output
The expected output will be the accuracy of the best model achieved after performing the grid search and the corresponding best hyperparameters. The output might look something like this:

```
Validation Accuracy with Best Parameters: 0.9856
Best Parameters: {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100}
```

### Explanation of Output Components:
- **Validation Accuracy**: This is the accuracy score achieved by the best model on the validation set after hyperparameter tuning.
- **Best Parameters**: These are the values of `n_estimators`, `learning_rate`, and `max_depth` that resulted in the highest cross-validated accuracy.

By using this approach, you can ensure that the gradient boosting model is optimized for better performance on your specific dataset.