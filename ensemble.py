# ensemble_model.py
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Import trained models from your existing scripts
from rf_model import rf, X_train, X_test, y_train, y_test
from ann_model import ann_pipeline
from svm_model import svm_grid


voting = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("svm", svm_grid.best_estimator_),
        ("ann", ann_pipeline)
    ],
    voting="soft"
)

voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
y_prob = voting.predict_proba(X_test)[:, 1]

print("\nSoft Voting Ensemble")
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

joblib.dump({"model": voting, "features": X_train.columns.tolist()}, "soft_voting_ensemble.pkl")
print("Soft Voting Ensemble saved as soft_voting_ensemble.pkl")

