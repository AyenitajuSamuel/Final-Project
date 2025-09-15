import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix, roc_curve, 
                           precision_recall_curve, auc)

# Import trained models and data
from rf_model import rf, X_test, y_test
from ann_model import ann_pipeline
from svm_model import svm_grid
from ensemble import voting # if you saved them in ensemble_model.py

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Collect models
models = {
    "Random Forest": rf,
    "SVM": svm_grid.best_estimator_,
    "ANN": ann_pipeline,
    "Soft Voting": voting
}

# Colors for consistent visualization
colors = ["#4F81BD", "#C0504D", "#9BBB59", "#8064A2"]

# Calculate all metrics for each model
all_metrics = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    predictions[name] = y_pred
    probabilities[name] = y_prob
    
    all_metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }

# 1. Comprehensive Bar Chart (All Metrics)
fig, ax = plt.subplots(figsize=(14, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
x = np.arange(len(metrics))
width = 0.2

for i, (model_name, metrics_dict) in enumerate(all_metrics.items()):
    values = [metrics_dict[metric] for metric in metrics]
    bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i], alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Performance Metrics', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics)
ax.legend(loc='upper right')
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Grouped Bar Chart (Accuracy vs AUC)
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(all_metrics.keys())
accuracy_scores = [all_metrics[name]['Accuracy'] for name in model_names]
auc_scores = [all_metrics[name]['AUC'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, accuracy_scores, width, label='Accuracy', color='skyblue', alpha=0.8)
bars2 = ax.bar(x + width/2, auc_scores, width, label='AUC', color='orange', alpha=0.8)

# Add value labels
for bar, value in zip(bars1, accuracy_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

for bar, value in zip(bars2, auc_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Models', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Accuracy vs AUC Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Combined ROC Curves
plt.figure(figsize=(10, 8))
for i, (name, model) in enumerate(models.items()):
    y_prob = probabilities[name]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=colors[i], lw=2, 
             label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('ROC Curves - All Models Comparison', fontsize=16, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Combined Precision-Recall Curves
plt.figure(figsize=(10, 8))
for i, (name, model) in enumerate(models.items()):
    y_prob = probabilities[name]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color=colors[i], lw=2,
             label=f'{name} (AUC = {pr_auc:.3f})')

# Baseline (random classifier for imbalanced data)
baseline = np.sum(y_test) / len(y_test)
plt.axhline(y=baseline, color='k', linestyle='--', 
            label=f'Random Classifier (AUC = {baseline:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Precision-Recall Curves - All Models Comparison', fontsize=16, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Confusion Matrices Grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (name, model) in enumerate(models.items()):
    y_pred = predictions[name]
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No CAD', 'CAD'], 
                yticklabels=['No CAD', 'CAD'],
                ax=axes[i])
    
    axes[i].set_xlabel('Predicted Label', fontweight='bold')
    axes[i].set_ylabel('True Label', fontweight='bold')
    axes[i].set_title(f'Confusion Matrix - {name}', fontweight='bold')

plt.tight_layout()
plt.show()

# 6. Feature Importance (Random Forest)
if hasattr(rf, 'feature_importances_'):
    feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature_{i}' for i in range(X_test.shape[1])]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importances)), importances[indices], color='forestgreen', alpha=0.8)
    plt.xlabel('Feature Index', fontweight='bold')
    plt.ylabel('Importance Score', fontweight='bold')
    plt.title('Random Forest - Feature Importance', fontsize=16, fontweight='bold')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 7. Model Performance Summary Table
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)
print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<10} {'AUC':<8}")
print("-"*80)
for name, metrics_dict in all_metrics.items():
    print(f"{name:<15} {metrics_dict['Accuracy']:<10.4f} {metrics_dict['Precision']:<11.4f} "
          f"{metrics_dict['Recall']:<8.4f} {metrics_dict['F1-Score']:<10.4f} {metrics_dict['AUC']:<8.4f}")
print("="*80)