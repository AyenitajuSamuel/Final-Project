import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report, precision_score, 
                           recall_score, f1_score, roc_curve, auc, precision_recall_curve)

# Import all models and data
from rf_model import rf, X_test, y_test
from ann_model import ann_pipeline
from svm_model import svm_grid
from ensemble import voting  # Assuming you have an ensemble model

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set1")

# Collect all models
models = {
    "Random Forest": rf,
    "SVM": svm_grid.best_estimator_,
    "ANN": ann_pipeline,
    "Soft Voting": voting
}

# Get predictions from all models
all_predictions = {}
all_probabilities = {}

for name, model in models.items():
    all_predictions[name] = model.predict(X_test)
    all_probabilities[name] = model.predict_proba(X_test)[:, 1]

# 1. Voting Contribution Analysis
def plot_voting_contributions():
    """Visualize how each model contributes to ensemble decisions"""
    
    # Get individual model probabilities for ensemble analysis
    rf_probs = rf.predict_proba(X_test)[:, 1]
    svm_probs = svm_grid.best_estimator_.predict_proba(X_test)[:, 1]
    ann_probs = ann_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate mean (soft voting equivalent)
    ensemble_probs = (rf_probs + svm_probs + ann_probs) / 3
    
    # Create subplots for different analyses
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Individual Model Probability Distributions
    axes[0, 0].hist(rf_probs, bins=30, alpha=0.6, label='Random Forest', density=True)
    axes[0, 0].hist(svm_probs, bins=30, alpha=0.6, label='SVM', density=True)
    axes[0, 0].hist(ann_probs, bins=30, alpha=0.6, label='ANN', density=True)
    axes[0, 0].hist(ensemble_probs, bins=30, alpha=0.8, label='Ensemble', 
                   color='black', histtype='step', linewidth=2, density=True)
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Model Probability Distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Correlation Matrix of Predictions
    prob_matrix = np.column_stack([rf_probs, svm_probs, ann_probs, ensemble_probs])
    corr_matrix = np.corrcoef(prob_matrix.T)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
               xticklabels=['RF', 'SVM', 'ANN', 'Ensemble'],
               yticklabels=['RF', 'SVM', 'ANN', 'Ensemble'],
               ax=axes[0, 1])
    axes[0, 1].set_title('Model Prediction Correlations')
    
    # Subplot 3: Agreement Analysis
    rf_pred = (rf_probs >= 0.5).astype(int)
    svm_pred = (svm_probs >= 0.5).astype(int)
    ann_pred = (ann_probs >= 0.5).astype(int)
    
    agreement_counts = []
    labels = []
    
    # Count different agreement patterns
    for rf_val in [0, 1]:
        for svm_val in [0, 1]:
            for ann_val in [0, 1]:
                mask = (rf_pred == rf_val) & (svm_pred == svm_val) & (ann_pred == ann_val)
                count = np.sum(mask)
                if count > 0:
                    agreement_counts.append(count)
                    labels.append(f'RF:{rf_val}, SVM:{svm_val}, ANN:{ann_val}')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(agreement_counts)))
    wedges, texts, autotexts = axes[1, 0].pie(agreement_counts, labels=labels, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
    axes[1, 0].set_title('Model Agreement Patterns')
    
    # Subplot 4: Ensemble vs Individual Performance
    ensemble_pred = voting.predict(X_test)
    
    models_for_comparison = ['RF', 'SVM', 'ANN', 'Ensemble']
    accuracies = []
    f1_scores_list = []
    
    for name, pred in zip(['RF', 'SVM', 'ANN'], [rf_pred, svm_pred, ann_pred]):
        accuracies.append((y_test == pred).mean())
        f1_scores_list.append(f1_score(y_test, pred))
    
    # Add ensemble
    accuracies.append((y_test == ensemble_pred).mean())
    f1_scores_list.append(f1_score(y_test, ensemble_pred))
    
    x = np.arange(len(models_for_comparison))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    bars2 = axes[1, 1].bar(x + width/2, f1_scores_list, width, label='F1-Score', alpha=0.8)
    
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Individual vs Ensemble Performance')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models_for_comparison)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

plot_voting_contributions()

# 2. Ensemble ROC and PR Curves Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

colors = ['blue', 'red', 'green', 'purple']
model_names = list(models.keys())

# ROC Curves
for i, (name, model) in enumerate(models.items()):
    y_prob = all_probabilities[name]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color=colors[i], lw=2, 
             label=f'{name} (AUC = {roc_auc:.3f})')

ax1.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random Classifier')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Receiver Operating Characteristic (ROC) Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# PR Curves
for i, (name, model) in enumerate(models.items()):
    y_prob = all_probabilities[name]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    ax2.plot(recall, precision, color=colors[i], lw=2, 
             label=f'{name} (AUC = {pr_auc:.3f})')

ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall (PR) Curves')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()