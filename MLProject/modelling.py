import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

mlflow.set_experiment("Heart Disease Classification")

def save_confusion_matrix(y_true, y_pred, filepath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(filepath)
    plt.close()

def main():
    mlflow.set_experiment("Heart Disease Classification")
    mlflow.sklearn.autolog(log_models=False)

    data_path = "heart_preprocessing.csv"  
    df = pd.read_csv(data_path)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    artifact_dir = "model"
    os.makedirs(artifact_dir, exist_ok=True)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        pred_probs = model.predict_proba(X_test)[:, 1]

        # Metrik utama
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, pred_probs)

        # Specificity
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)

        # Log metrik manual
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("specificity", specificity)

        # Simpan dan log confusion matrix
        cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
        save_confusion_matrix(y_test, preds, cm_path)
        mlflow.log_artifact(cm_path)

        # Simpan dan log model manual (joblib)
        model_pkl_path = os.path.join(artifact_dir, "model.pkl")
        joblib.dump(model, model_pkl_path)
        mlflow.log_artifact(model_pkl_path, artifact_path="model")

        # Buat input example dan infer signature
        input_example = X_train.head(5)
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

    print("Training selesai dan model berhasil tercatat di MLflow.")

if __name__ == "__main__":
    main()
