It seems like the code was successfully executed, and the results were printed to the console. However, since I can't display the output here, let me summarize what you should expect:

1. **Model Evaluation**:
   - You will see the accuracy and F1-score for each model (Logistic Regression, Random Forest, and Gradient Boosting).
   - The best-performing model among these three will be indicated.

2. **Best RandomForestClassifier**:
   - You will also get the performance metrics (accuracy and F1-score) for the best RandomForestClassifier found by GridSearchCV.
   - Additionally, you will see the optimal hyperparameters for this best model.

Here's a hypothetical example of what the output might look like:

```
Logistic Regression:
  Accuracy: 0.8571
  F1 Score: 0.9234

Random Forest:
  Accuracy: 0.9123
  F1 Score: 0.9456

Gradient Boosting:
  Accuracy: 0.9078
  F1 Score: 0.9354

Best RandomForestClassifier:
  Accuracy: 0.9123
  F1 Score: 0.9456
  Best Parameters: {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100}
```

In this example:
- The Random Forest model is the best-performing with an accuracy of 0.9123 and F1-score of 0.9456.
- The optimal hyperparameters for the best RandomForestClassifier are `max_depth=None`, `min_samples_split=5`, and `n_estimators=100`.

Make sure to replace the hypothetical values with the actual output from your execution.