```json
{
  "id": "call_3",
  "type": "function",
  "function": {
    "name": "python_interpreter",
    "arguments": "from sklearn.metrics import classification_report, roc_auc_score\n\n# Make predictions on the test set\ny_pred = model.predict(X_test_scaled)\ny_pred_proba = model.predict_proba(X_test_scaled)[:, 1]\n\n# Calculate metrics\naccuracy = accuracy_score(y_test, y_pred)\nprecision = precision_score(y_test, y_pred)\nrecall = recall_score(y_test, y_pred)\nf1_score = f1_score(y_test, y_pred)\nroc_auc = roc_auc_score(y_test, y_pred_proba)\n\nprint(f\"Accuracy: {accuracy}\")\nprint(f\"Precision: {precision}\")\nprint(f\"Recall: {recall}\")\nprint(f\"F1-Score: {f1_score}\")\nprint(f\"AUC-ROC: {roc_auc}\")\n\n# Classification report\nprint(classification_report(y_test, y_pred))"
  }
}
```