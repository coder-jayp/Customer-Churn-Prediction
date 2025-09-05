import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

df = pd.read_csv("data/processed/churn_processed.csv")

drop_cols = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long",
    "Latitude", "Longitude", "Churn Label", "Churn Value", "Churn Score",
    "CLTV", "Churn Reason"
]

X = df.drop(columns=drop_cols, errors="ignore")
y = df["Churn Value"]

X_train, X_test, y_train, y_test = train_test_split(
    X.astype(np.float32), y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(
            eval_metric="logloss",
            tree_method="hist",   
            device="cuda",      
            predictor="auto",
            n_jobs=-1
        ),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5, 7]
        }
    },
    "LightGBM": {
        "model": lgb.LGBMClassifier(
            device="gpu",         
            gpu_platform_id=0,  
            gpu_device_id=0
        ),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [-1, 5, 10]
        }
    }
}

results = []

for name, mp in models.items():
    print(f"\n🔹 Training {name}...")
    clf = GridSearchCV(mp["model"], mp["params"], cv=3, scoring="roc_auc", n_jobs=-1)
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Best Params": clf.best_params_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

results_df = pd.DataFrame(results)
print("\n📊 Model Performance Comparison")
print(results_df)

results_df.to_csv("models/model_results.csv", index=False)

best_model_name = results_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]["Model"]
print(f"\n✅ Best Model: {best_model_name}")

final_model = models[best_model_name]["model"].set_params(
    **results_df.loc[results_df['Model'] == best_model_name, 'Best Params'].values[0]
)
final_model.fit(X_train, y_train)
joblib.dump(final_model, "models/best_model.pkl")

if best_model_name in ["RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]:
    importances = final_model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp)
    plt.title(f"Feature Importance - {best_model_name}")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    plt.close()

explainer = shap.Explainer(final_model, X_train)
shap_values = explainer(X_test, check_additivity=False)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("models/shap_summary.png")
plt.close()

print("\n✅ Training complete. Best model saved in 'models/best_model.pkl'")
