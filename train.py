"""Train the predictive-maintenance model (same pipeline as the notebook) and save model.joblib."""
import warnings, json
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, average_precision_score
from features import add_features, NUM_FEATURES, ALL_NUM, CAT

warnings.filterwarnings("ignore")
RS = 42

def load(path="data.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig").rename(columns={
        "Air temperature [K]": "air_temp", "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm", "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear", "Machine failure": "machine_failure"})
    return df

def make_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), ALL_NUM),
        ("cat", OneHotEncoder(categories=[["L", "M", "H"]], drop="first", sparse_output=False), CAT)])

def main():
    df = add_features(load())
    X, y = df[ALL_NUM + CAT], df["machine_failure"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=RS),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=RS),
        "HistGradientBoosting": HistGradientBoostingClassifier(class_weight="balanced", learning_rate=0.08, max_iter=300, random_state=RS),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="aucpr", random_state=RS, n_jobs=-1)
    except ImportError: pass
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    scores = {}
    for n, m in models.items():
        pipe = Pipeline([("pre", make_pre()), ("clf", m)])
        scores[n] = cross_validate(pipe, Xtr, ytr, cv=cv, scoring="average_precision", n_jobs=1)["test_score"].mean()
        print(f"{n:22s} PR-AUC={scores[n]:.4f}")
    best = max(scores, key=scores.get)
    pipe = Pipeline([("pre", make_pre()), ("clf", models[best])])
    oof = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(ytr, oof)
    thr = float(t[np.argmax(2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9))])
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]; pred = (proba >= thr).astype(int)
    metrics = {"model": best, "threshold": round(thr, 3),
               "precision": round(precision_score(yte, pred), 3), "recall": round(recall_score(yte, pred), 3),
               "f1": round(f1_score(yte, pred), 3), "pr_auc": round(average_precision_score(yte, proba), 3)}
    print(metrics)
    joblib.dump({"pipeline": pipe, "threshold": thr, "metrics": metrics}, "model.joblib")
    json.dump(metrics, open("metrics.json", "w"), indent=1)

if __name__ == "__main__":
    main()
