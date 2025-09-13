import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from rf_model import rf, X_test, y_test

# Predict on test set
y_pred_rf = rf.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Plot confusion matrix with seaborn heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Random Forest")

# Compute metrics
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)

# Add metrics as text on the plot
plt.text(2.5, 0.5, f"Precision: {precision:.2f}", fontsize=10, color="black", va="center")
plt.text(2.5, 1.0, f"Recall: {recall:.2f}", fontsize=10, color="black", va="center")
plt.text(2.5, 1.5, f"F1-score: {f1:.2f}", fontsize=10, color="black", va="center")

plt.tight_layout()
plt.show()

# Print classification report as well
print(classification_report(y_test, y_pred_rf))
