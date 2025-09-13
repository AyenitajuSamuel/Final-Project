import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Import trained models and data
from rf_model import rf, X_test, y_test
from ann_model import ann_pipeline
from svm_model import svm_grid
from ensemble import voting # if you saved them in ensemble_model.py

# Collect models
models = {
    "Random Forest": rf,
    "SVM": svm_grid.best_estimator_,
    "ANN": ann_pipeline,
    "Soft Voting": voting
}

# Evaluate ROC-AUC for each model
scores = {}
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    scores[name] = roc_auc_score(y_test, y_prob)

# Plot bar chart
plt.figure(figsize=(8,6))
plt.bar(scores.keys(), scores.values(), color=["#4F81BD", "#C0504D", "#9BBB59", "#8064A2", "#F79646"])
plt.ylabel("ROC-AUC Score")
plt.ylim(0, 1)
plt.title("Model Performance Comparison (ROC-AUC)")
plt.xticks(rotation=30, ha="right")

# Annotate bars with values
for i, (name, value) in enumerate(scores.items()):
    plt.text(i, value + 0.02, f"{value:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.show()
