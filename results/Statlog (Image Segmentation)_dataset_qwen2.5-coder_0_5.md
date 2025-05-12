It seems like you're trying to evaluate machine learning models using different algorithms and compare their performance. However, I can't run code here, so I'll guide you through how to do it step-by-step.

1. **Reading the Dataset**: Ensure that you have your dataset correctly loaded into a pandas DataFrame. Typically, this would look like:
   ```python
   features, targets = read_dataset("./datasets/Statlog (Image Segmentation)/")
   ```

2. **Data Preprocessing**: Make sure to handle any missing values and normalize or standardize the features if necessary.
   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   features_scaled = scaler.fit_transform(features)
   ```

3. **Splitting the Data**: Divide your dataset into training and testing sets to evaluate the model's performance on unseen data.
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(features_scaled, targets, test_size=0.2, random_state=42)
   ```

4. **Evaluating Models**: Define a dictionary of models you want to evaluate and iterate over it to fit each model on the training data and evaluate it on the testing set.
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.svm import SVC
   from sklearn.ensemble import RandomForestClassifier

   models = {
       'Logistic Regression': LogisticRegression(),
       'SVM': SVC(probability=True),
       'Random Forest': RandomForestClassifier()
   }

   for name, model in models.items():
       model.fit(X_train, y_train)
       scores = cross_val_score(model, X_train, y_train, cv=5)
       print(f'{name}: {scores.mean():.4f} (+/- {scores.std():.4f})')
   ```

5. **Choosing the Best Model**: Based on the average performance across the different folds of cross-validation, choose the model that performs best.

6. **Evaluating on Test Set**: After selecting the best model, evaluate it on the test set to get an unbiased estimate of its performance.
   ```python
   from sklearn.metrics import classification_report

   y_pred = best_model.predict(X_test)
   print(classification_report(y_test, y_pred))
   ```

By following these steps, you should be able to evaluate different machine learning models for your image segmentation task and select the one that performs the best.