from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
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

# ANN pipeline (scales data first, then trains neural net)
ann_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ann", MLPClassifier(hidden_layer_sizes=(64, 32),
                          activation="relu",
                          solver="adam",
                          alpha=1e-4,        # L2 regularization
                          max_iter=1000,
                          early_stopping=True,
                          random_state=42))
])

# Train ANN
ann_pipeline.fit(X_train, y_train)

# Predictions
y_pred = ann_pipeline.predict(X_test)
y_prob = ann_pipeline.predict_proba(X_test)[:, 1]

print("ROC-AUC (ANN):", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))
