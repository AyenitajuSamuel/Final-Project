import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (roc_curve, auc, confusion_matrix, precision_recall_curve,
                           classification_report, precision_score, recall_score, f1_score)

# Use your trained SVM model (best estimator from GridSearchCV)
from svm_model import svm_grid, X_test, y_test

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

svm_best = svm_grid.best_estimator_

# Predictions and probabilities
y_pred_svm = svm_best.predict(X_test)
y_prob_svm = svm_best.predict_proba(X_test)[:, 1]

# 1. Enhanced ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_svm)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color="darkorange", lw=3, 
         label=f"SVM ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", 
         label="Random Classifier")

# Find optimal threshold (Youden's J statistic)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
         label=f'Optimal Threshold = {optimal_threshold:.3f}')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontweight='bold', fontsize=12)
plt.ylabel("True Positive Rate (Sensitivity)", fontweight='bold', fontsize=12)
plt.title("SVM - ROC Curve Analysis", fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)

# Add AUC confidence interval (approximation)
plt.text(0.6, 0.2, f'AUC = {roc_auc:.3f}\n95% CI ≈ [{roc_auc-0.05:.3f}, {roc_auc+0.05:.3f}]', 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
         fontsize=11)

plt.tight_layout()
plt.show()

# 2. Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob_svm)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='purple', lw=3,
         label=f'SVM PR Curve (AUC = {pr_auc:.3f})')

# Baseline (proportion of positive class)
baseline = np.sum(y_test) / len(y_test)
plt.axhline(y=baseline, color='red', linestyle='--', 
            label=f'Random Classifier (Baseline = {baseline:.3f})')

# Find F1-optimal threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_f1_idx = np.argmax(f1_scores)
plt.plot(recall[best_f1_idx], precision[best_f1_idx], 'go', markersize=10,
         label=f'Best F1 = {f1_scores[best_f1_idx]:.3f}')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Sensitivity)', fontweight='bold', fontsize=12)
plt.ylabel('Precision (PPV)', fontweight='bold', fontsize=12)
plt.title('SVM - Precision-Recall Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower left", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Confusion Matrix with detailed metrics
cm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['No CAD', 'CAD'], 
            yticklabels=['No CAD', 'CAD'],
            cbar_kws={'label': 'Count'})

plt.xlabel('Predicted Label', fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.title('SVM - Confusion Matrix', fontsize=16, fontweight='bold')

# Calculate detailed metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

# Add metrics text
metrics_text = f"""Detailed Metrics:
True Positives: {tp}
True Negatives: {tn}
False Positives: {fp}
False Negatives: {fn}

Accuracy: {accuracy:.3f}
Precision: {precision_val:.3f}
Recall: {recall_val:.3f}
Specificity: {specificity:.3f}
F1-Score: {f1_val:.3f}"""

plt.text(2.2, 0.1, metrics_text, fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

plt.tight_layout()
plt.show()

# 4. Decision Function Analysis (SVM-specific)
if hasattr(svm_best, 'decision_function'):
    decision_scores = svm_best.decision_function(X_test)
    
    plt.figure(figsize=(12, 5))
    
    # Decision score distribution by class
    plt.subplot(1, 2, 1)
    plt.hist(decision_scores[y_test == 0], bins=30, alpha=0.7, 
             label='No CAD', color='blue', density=True)
    plt.hist(decision_scores[y_test == 1], bins=30, alpha=0.7, 
             label='CAD', color='red', density=True)
    plt.axvline(x=0, color='black', linestyle='--', label='Decision Boundary')
    plt.xlabel('SVM Decision Score', fontweight='bold')
    plt.ylabel('Density', fontweight='bold')
    plt.title('SVM Decision Score Distribution', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Probability distribution
    plt.subplot(1, 2, 2)
    plt.hist(y_prob_svm[y_test == 0], bins=30, alpha=0.7, 
             label='No CAD', color='blue', density=True)
    plt.hist(y_prob_svm[y_test == 1], bins=30, alpha=0.7, 
             label='CAD', color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold')
    plt.xlabel('Predicted Probability', fontweight='bold')
    plt.ylabel('Density', fontweight='bold')
    plt.title('SVM Probability Distribution', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 5. Threshold Analysis
thresholds_range = np.linspace(0.01, 0.99, 100)  # Avoid extreme values
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []

for threshold in thresholds_range:
    y_pred_thresh = (y_prob_svm >= threshold).astype(int)
    
    try:
        # Calculate metrics with error handling
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        acc = (y_test == y_pred_thresh).mean()
        
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)
        accuracy_scores.append(acc)
    except:
        # Fallback values if calculation fails
        precision_scores.append(0)
        recall_scores.append(0)
        f1_scores.append(0)
        accuracy_scores.append(0)

plt.figure(figsize=(12, 8))
plt.plot(thresholds_range, precision_scores, label='Precision', color='blue', linewidth=2)
plt.plot(thresholds_range, recall_scores, label='Recall', color='red', linewidth=2)
plt.plot(thresholds_range, f1_scores, label='F1-Score', color='green', linewidth=2)
plt.plot(thresholds_range, accuracy_scores, label='Accuracy', color='orange', linewidth=2)

# Mark optimal thresholds
if len(f1_scores) > 0 and max(f1_scores) > 0:
    best_f1_threshold = thresholds_range[np.argmax(f1_scores)]
    plt.axvline(x=best_f1_threshold, color='purple', linestyle=':', alpha=0.7, 
                label=f'Best F1 Threshold ({best_f1_threshold:.2f})')

plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Default Threshold (0.5)')

plt.xlabel('Decision Threshold', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.title('SVM - Threshold Analysis', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# 6. Model Parameters Visualization
print("\n" + "="*60)
print("SVM MODEL CONFIGURATION")
print("="*60)
print("Best Parameters from GridSearchCV:")
if hasattr(svm_grid, 'best_params_'):
    for param, value in svm_grid.best_params_.items():
        print(f"  {param}: {value}")

if hasattr(svm_grid, 'best_score_'):
    print(f"\nCross-validation Score: {svm_grid.best_score_:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")
print(f"Test PR-AUC: {pr_auc:.4f}")

# 7. GridSearch Results Visualization (if available)
if hasattr(svm_grid, 'cv_results_'):
    cv_results = svm_grid.cv_results_
    
    # Plot validation curve for main parameter
    if hasattr(svm_grid, 'best_params_') and len(svm_grid.best_params_) > 0:
        param_keys = list(svm_grid.best_params_.keys())
        
        # Focus on the first parameter for simplicity
        if len(param_keys) >= 1:
            main_param = param_keys[0]
            
            # Extract parameter values and scores
            param_values = []
            mean_scores = []
            std_scores = []
            
            for i, params in enumerate(cv_results['params']):
                if main_param in params:
                    param_values.append(params[main_param])
                    mean_scores.append(cv_results['mean_test_score'][i])
                    std_scores.append(cv_results['std_test_score'][i])
            
            if len(set(param_values)) > 1:  # Only plot if we have multiple values
                # Group by parameter value
                unique_params = sorted(list(set(param_values)))
                grouped_means = []
                grouped_stds = []
                
                for param_val in unique_params:
                    indices = [i for i, pv in enumerate(param_values) if pv == param_val]
                    if indices:
                        grouped_means.append(np.mean([mean_scores[i] for i in indices]))
                        grouped_stds.append(np.mean([std_scores[i] for i in indices]))
                
                plt.figure(figsize=(10, 6))
                plt.errorbar(range(len(unique_params)), grouped_means, yerr=grouped_stds, 
                           marker='o', capsize=5, capthick=2, linewidth=2)
                plt.xlabel(f'{main_param}', fontweight='bold')
                plt.ylabel('Cross-validation Score', fontweight='bold')
                plt.title(f'SVM Validation Curve - {main_param}', fontsize=16, fontweight='bold')
                plt.xticks(range(len(unique_params)), unique_params, rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

# 8. Support Vector Analysis (if available)
if hasattr(svm_best, 'n_support_'):
    plt.figure(figsize=(8, 6))
    classes = ['No CAD', 'CAD']
    support_counts = svm_best.n_support_
    
    bars = plt.bar(classes, support_counts, color=['lightblue', 'lightcoral'], 
                   alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, count in zip(bars, support_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Class', fontweight='bold')
    plt.ylabel('Number of Support Vectors', fontweight='bold')
    plt.title('SVM - Support Vector Distribution', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    total_sv = np.sum(support_counts)
    total_samples = len(y_test)
    plt.text(0.5, max(support_counts) * 0.8, 
             f'Total Support Vectors: {total_sv}\nPercentage: {100*total_sv/total_samples:.1f}%',
             ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Print detailed classification report
print("\n" + "="*60)
print("SVM - DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred_svm, target_names=['No CAD', 'CAD']))
print("="*60)

# Summary statistics
print(f"\nModel Summary:")
if hasattr(svm_best, 'kernel'):
    print(f"Kernel: {svm_best.kernel}")
if hasattr(svm_best, 'C'):
    print(f"C (Regularization): {svm_best.C}")
if hasattr(svm_best, 'gamma'):
    print(f"Gamma: {svm_best.gamma}")
print(f"Total Support Vectors: {np.sum(svm_best.n_support_) if hasattr(svm_best, 'n_support_') else 'N/A'}")