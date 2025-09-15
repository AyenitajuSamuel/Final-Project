import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report, precision_score, 
                           recall_score, f1_score, roc_curve, auc, precision_recall_curve)
from ann_model import ann_pipeline, X_test, y_test

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("plasma")

# Predictions and probabilities
y_pred_ann = ann_pipeline.predict(X_test)
y_prob_ann = ann_pipeline.predict_proba(X_test)[:, 1]

# 1. Enhanced Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_ann)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", 
            xticklabels=['No CAD', 'CAD'], yticklabels=['No CAD', 'CAD'],
            cbar_kws={'label': 'Count'})
plt.xlabel("Predicted Label", fontweight='bold')
plt.ylabel("True Label", fontweight='bold')
plt.title("Artificial Neural Network - Confusion Matrix", fontsize=16, fontweight='bold')

# Compute detailed metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Add metrics as text box
metrics_text = f"""Performance Metrics:
Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall/Sensitivity: {recall:.3f}
Specificity: {specificity:.3f}
F1-Score: {f1:.3f}

Confusion Matrix:
TP: {tp}, TN: {tn}
FP: {fp}, FN: {fn}"""

plt.text(2.2, 0.1, metrics_text, fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lavender", alpha=0.9))

plt.tight_layout()
plt.show()

# 2. ROC Curve Analysis
fpr, tpr, thresholds = roc_curve(y_test, y_prob_ann)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkmagenta', lw=3, 
         label=f'ANN ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Classifier')

# Find optimal threshold using Youden's J statistic
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
         label=f'Optimal Threshold = {optimal_threshold:.3f}')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold', fontsize=12)
plt.title('ANN - ROC Curve Analysis', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Precision-Recall Curve
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_prob_ann)
pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='darkgreen', lw=3,
         label=f'ANN PR Curve (AUC = {pr_auc:.3f})')

# Baseline for imbalanced dataset
baseline = np.sum(y_test) / len(y_test)
plt.axhline(y=baseline, color='red', linestyle='--', 
            label=f'Random Classifier (Baseline = {baseline:.3f})')

# Find best F1 point
f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
best_f1_idx = np.argmax(f1_scores)
plt.plot(recall_curve[best_f1_idx], precision_curve[best_f1_idx], 'go', 
         markersize=10, label=f'Best F1 = {f1_scores[best_f1_idx]:.3f}')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontweight='bold', fontsize=12)
plt.ylabel('Precision', fontweight='bold', fontsize=12)
plt.title('ANN - Precision-Recall Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower left", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Training Loss Curves (if available)
ann_model = ann_pipeline.named_steps['ann']
if hasattr(ann_model, 'loss_curve_'):
    plt.figure(figsize=(12, 5))
    
    # Training loss
    plt.subplot(1, 2, 1)
    plt.plot(ann_model.loss_curve_, color='blue', linewidth=2, label='Training Loss')
    if hasattr(ann_model, 'validation_scores_'):
        plt.plot(ann_model.validation_scores_, color='red', linewidth=2, label='Validation Score')
    plt.xlabel('Iterations', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.title('ANN Training Progress', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate analysis (if adaptive)
    plt.subplot(1, 2, 2)
    if hasattr(ann_model, 'learning_rate_') and ann_model.learning_rate_init != ann_model.learning_rate_:
        lr_history = [ann_model.learning_rate_init * (0.1 ** (i // 10)) for i in range(len(ann_model.loss_curve_))]
        plt.plot(lr_history, color='orange', linewidth=2)
        plt.xlabel('Iterations', fontweight='bold')
        plt.ylabel('Learning Rate', fontweight='bold')
        plt.title('Learning Rate Schedule', fontweight='bold')
        plt.yscale('log')
    else:
        # Plot convergence rate
        if len(ann_model.loss_curve_) > 10:
            loss_diff = np.diff(ann_model.loss_curve_)
            plt.plot(loss_diff, color='purple', linewidth=2)
            plt.xlabel('Iterations', fontweight='bold')
            plt.ylabel('Loss Change', fontweight='bold')
            plt.title('Convergence Rate', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'Insufficient data\nfor convergence plot', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Convergence Analysis', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 5. Network Architecture Visualization
def plot_network_architecture():
    """Visualize the neural network architecture"""
    layers = ann_model.hidden_layer_sizes
    if isinstance(layers, int):
        layers = (layers,)
    
    # Add input and output layers
    all_layers = [X_test.shape[1]] + list(layers) + [1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate positions
    max_neurons = max(all_layers)
    layer_positions = np.linspace(0, 10, len(all_layers))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_layers)))
    
    for i, (layer_size, x_pos) in enumerate(zip(all_layers, layer_positions)):
        y_positions = np.linspace(0, max_neurons, layer_size)
        
        # Draw neurons
        for y_pos in y_positions:
            circle = plt.Circle((x_pos, y_pos), 0.3, color=colors[i], alpha=0.7)
            ax.add_patch(circle)
        
        # Add layer labels
        if i == 0:
            ax.text(x_pos, max_neurons + 1, f'Input\n({layer_size} features)', 
                   ha='center', va='bottom', fontweight='bold')
        elif i == len(all_layers) - 1:
            ax.text(x_pos, max_neurons + 1, f'Output\n({layer_size} class)', 
                   ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(x_pos, max_neurons + 1, f'Hidden {i}\n({layer_size} neurons)', 
                   ha='center', va='bottom', fontweight='bold')
    
    # Draw connections (simplified - just show layer-to-layer connections)
    for i in range(len(layer_positions) - 1):
        x1, x2 = layer_positions[i], layer_positions[i + 1]
        y1_positions = np.linspace(0, max_neurons, all_layers[i])
        y2_positions = np.linspace(0, max_neurons, all_layers[i + 1])
        
        for y1 in y1_positions:
            for y2 in y2_positions:
                ax.plot([x1 + 0.3, x2 - 0.3], [y1, y2], 'k-', alpha=0.1, linewidth=0.5)
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-2, max_neurons + 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=16, fontweight='bold')
    
    # Add activation function info
    info_text = f"""Network Configuration:
    Activation: {ann_model.activation}
    Solver: {ann_model.solver}
    Learning Rate: {ann_model.learning_rate_init}
    Alpha (L2): {ann_model.alpha}
    Max Iterations: {ann_model.max_iter}
    Converged: {'Yes' if ann_model.n_iter_ < ann_model.max_iter else 'No'}
    Final Iterations: {ann_model.n_iter_}"""
    
    ax.text(11.5, max_neurons/2, info_text, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.9))
    
    plt.tight_layout()
    plt.show()

plot_network_architecture()

# 6. Probability Distribution Analysis
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_prob_ann[y_test == 0], bins=30, alpha=0.7, 
         label='No CAD', color='blue', density=True)
plt.hist(y_prob_ann[y_test == 1], bins=30, alpha=0.7, 
         label='CAD', color='red', density=True)
plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
plt.xlabel('Predicted Probability', fontweight='bold')
plt.ylabel('Density', fontweight='bold')
plt.title('ANN Probability Distribution by Class', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Calibration plot
plt.subplot(1, 2, 2)
from sklearn.calibration import calibration_curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob_ann, n_bins=10)

plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
         color='darkred', linewidth=2, label='ANN')
plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
plt.xlabel('Mean Predicted Probability', fontweight='bold')
plt.ylabel('Fraction of Positives', fontweight='bold')
plt.title('Calibration Plot', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Threshold Analysis
thresholds_range = np.linspace(0, 1, 100)
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []

for threshold in thresholds_range:
    y_pred_thresh = (y_prob_ann >= threshold).astype(int)
    
    if np.sum(y_pred_thresh) > 0 and np.sum(y_pred_thresh) < len(y_pred_thresh):
        precision_scores.append(precision_score(y_test, y_pred_thresh))
        recall_scores.append(recall_score(y_test, y_pred_thresh))
        f1_scores.append(f1_score(y_test, y_pred_thresh))
        accuracy_scores.append((y_test == y_pred_thresh).mean())
    else:
        precision_scores.append(0 if np.sum(y_pred_thresh) == 0 else 1)
        recall_scores.append(0)
        f1_scores.append(0)
        accuracy_scores.append(max((y_test == 0).mean(), (y_test == 1).mean()))

plt.figure(figsize=(12, 8))
plt.plot(thresholds_range, precision_scores, label='Precision', color='blue', linewidth=2)
plt.plot(thresholds_range, recall_scores, label='Recall', color='red', linewidth=2)
plt.plot(thresholds_range, f1_scores, label='F1-Score', color='green', linewidth=2)
plt.plot(thresholds_range, accuracy_scores, label='Accuracy', color='orange', linewidth=2)

# Mark optimal thresholds
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds_range[best_f1_idx]
plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Default Threshold (0.5)')
plt.axvline(x=best_f1_threshold, color='purple', linestyle=':', alpha=0.7, 
            label=f'Best F1 Threshold ({best_f1_threshold:.2f})')

plt.xlabel('Decision Threshold', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.title('ANN - Threshold Analysis', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# Print detailed results
print("\n" + "="*60)
print("ARTIFICIAL NEURAL NETWORK - DETAILED ANALYSIS")
print("="*60)
print("Network Architecture:")
print(f"  Input features: {X_test.shape[1]}")
print(f"  Hidden layers: {ann_model.hidden_layer_sizes}")
print(f"  Output: 1 (binary classification)")
print(f"  Total parameters: ~{sum(ann_model.coefs_[i].size for i in range(len(ann_model.coefs_))) + sum(ann_model.intercepts_[i].size for i in range(len(ann_model.intercepts_)))}")

print(f"\nTraining Details:")
print(f"  Iterations: {ann_model.n_iter_}")
print(f"  Final loss: {ann_model.loss_:.6f}")
print(f"  Solver: {ann_model.solver}")
print(f"  Learning rate: {ann_model.learning_rate_init}")
print(f"  Regularization (alpha): {ann_model.alpha}")

print(f"\nPerformance Metrics:")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  PR-AUC: {pr_auc:.4f}")
print(f"  Best F1 Threshold: {best_f1_threshold:.3f}")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred_ann, target_names=['No CAD', 'CAD']))
print("="*60)