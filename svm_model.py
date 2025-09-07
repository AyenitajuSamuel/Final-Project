from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from dataset import dataset



# Assume target column is named 'target'
X = dataset.drop(columns=["target"])
y = dataset["target"]

# Example split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Pipeline: scaling + SVM
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
])

# Hyperparameter grid
svm_params = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.01, 0.001]
}

# Grid Search
svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=5, scoring="roc_auc", n_jobs=-1)
svm_grid.fit(X_train, y_train)

y_pred = svm_grid.predict(X_test)
y_prob = svm_grid.predict_proba(X_test)[:,1]

print("Best Params (SVM):", svm_grid.best_params_)
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))
