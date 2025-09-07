from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from dataset import dataset


if "num" in dataset.columns:
    dataset["target"] = dataset["num"].apply(lambda v: 1 if v > 0 else 0)
    dataset = dataset.drop(columns=["num"])

# Features and labels
X = dataset.drop(columns=["target"])
y = dataset["target"]

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(n_estimators=600,
                            max_depth=None,
                            min_samples_leaf=3,
                            class_weight="balanced",
                            n_jobs=-1,
                            random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

print("ROC-AUC (RF):", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# Feature importance
import pandas as pd
# Save feature names from the dataset
feature_names = X.columns

# Get feature importances
feat_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(feat_imp.head(10))
