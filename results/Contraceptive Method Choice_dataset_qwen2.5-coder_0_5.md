```py
print("Final Accuracy:", accuracy)
```

This code will print the final accuracy of the model on the test set. The output will be a single floating-point number representing the accuracy of the model in predicting the contraceptive method choice based on the given features.

To summarize, the steps we followed were:
1. Import necessary libraries.
2. Read and preprocess the dataset.
3. Split the data into training and testing sets.
4. Create a pipeline with preprocessing and a classification model (RandomForestClassifier).
5. Train the model on the training data.
6. Evaluate the model's performance on the test set.

The final accuracy provides a measure of how well our model generalizes to new, unseen data from the same population.