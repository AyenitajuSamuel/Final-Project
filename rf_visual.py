import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report, precision_score, 
                           recall_score, f1_score, roc_curve, auc, precision_recall_curve)
from rf_model import rf, X_test, y_test

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Predict on test set
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# 1. Enhanced Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['No CAD', 'CAD'], yticklabels=['No CAD', 'CAD'],
            cbar_kws={'label': 'Count'})
plt.xlabel("Predicted Label", fontweight='bold')
plt.ylabel("True Label", fontweight='bold')
plt.title("Random Forest - Confusion Matrix", fontsize=16, fontweight='bold')

# Compute detailed metrics
accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])

# Add metrics as text box
metrics_text = f"""Performance Metrics:
Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall/Sensitivity: {recall:.3f}
Specificity: {specificity:.3f}
F1-Score: {f1:.3f}"""

plt.text(2.7, 0.2, metrics_text, fontsize=11, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.show()

# 2. ROC Curve for Random Forest
fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'Random Forest ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
plt.title('Random Forest - ROC Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_rf)
pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='green', lw=2,
         label=f'Random Forest PR (AUC = {pr_auc:.3f})')

# Baseline for imbalanced dataset
baseline = np.sum(y_test) / len(y_test)
plt.axhline(y=baseline, color='red', linestyle='--', 
            label=f'Random Classifier (Baseline = {baseline:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Random Forest - Precision-Recall Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Feature Importance Analysis
if hasattr(rf, 'feature_importances_'):
    feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature_{i}' for i in range(X_test.shape[1])]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Top 15 most important features
    top_n = min(15, len(importances))
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(top_n), importances[indices[:top_n]], 
                   color='forestgreen', alpha=0.7, edgecolor='darkgreen')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{importances[indices[i]]:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Features', fontweight='bold')
    plt.ylabel('Importance Score', fontweight='bold')
    plt.title(f'Random Forest - Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 5. Tree Depth and Estimators Analysis (if available)
if hasattr(rf, 'estimators_'):
    n_estimators = len(rf.estimators_)
    depths = [tree.tree_.max_depth for tree in rf.estimators_]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of tree depths
    ax1.hist(depths, bins=15, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_xlabel('Tree Depth', fontweight='bold')
    ax1.set_ylabel('Number of Trees', fontweight='bold')
    ax1.set_title('Distribution of Tree Depths in Random Forest', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Feature importance cumulative plot
    sorted_importances = np.sort(importances)[::-1]
    cumsum_importances = np.cumsum(sorted_importances)
    
    ax2.plot(range(1, len(cumsum_importances) + 1), cumsum_importances, 
             'o-', color='red', linewidth=2, markersize=4)
    ax2.axhline(y=0.8, color='green', linestyle='--', label='80% Threshold')
    ax2.axhline(y=0.9, color='orange', linestyle='--', label='90% Threshold')
    ax2.set_xlabel('Number of Features', fontweight='bold')
    ax2.set_ylabel('Cumulative Feature Importance', fontweight='bold')
    ax2.set_title('Cumulative Feature Importance', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 6. Decision Threshold Analysis
thresholds_range = np.linspace(0, 1, 100)
precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds_range:
    y_pred_thresh = (y_prob_rf >= threshold).astype(int)
    
    # Avoid division by zero
    if np.sum(y_pred_thresh) > 0:
        precision_scores.append(precision_score(y_test, y_pred_thresh))
        recall_scores.append(recall_score(y_test, y_pred_thresh))
        f1_scores.append(f1_score(y_test, y_pred_thresh))
    else:
        precision_scores.append(0)
        recall_scores.append(0)
        f1_scores.append(0)

plt.figure(figsize=(10, 6))
plt.plot(thresholds_range, precision_scores, label='Precision', color='blue', linewidth=2)
plt.plot(thresholds_range, recall_scores, label='Recall', color='red', linewidth=2)
plt.plot(thresholds_range, f1_scores, label='F1-Score', color='green', linewidth=2)

# Mark the default threshold (0.5)
plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Default Threshold (0.5)')

plt.xlabel('Decision Threshold', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.title('Random Forest - Threshold Analysis', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# Print detailed classification report
print("\n" + "="*60)
print("RANDOM FOREST - DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred_rf, target_names=['No CAD', 'CAD']))
print("="*60)

# Additional metrics summary
print(f"\nModel Configuration:")
print(f"Number of trees: {rf.n_estimators}")
print(f"Max depth: {rf.max_depth}")
print(f"Min samples split: {rf.min_samples_split}")
print(f"Min samples leaf: {rf.min_samples_leaf}")
print(f"\nPerformance Summary:")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Balanced Accuracy: {(sensitivity + specificity) / 2:.4f}")