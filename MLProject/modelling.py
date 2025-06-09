import os
import json
import shutil
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
import dagshub

# Inisialisasi dagshub dan mlflow
dagshub.init(repo_owner='daffaakifah', repo_name='Membangun_model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/daffaakifah/Membangun_model.mlflow/")
mlflow.set_experiment("Heart Disease Classification")
mlflow.autolog()

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
    data_path = "Membangun_model/heart_preprocessing.csv"  # Pastikan path ini benar sesuai lingkungan Anda
    df = pd.read_csv(data_path)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    artifact_dir = "model"
    os.makedirs(artifact_dir, exist_ok=True)

    with mlflow.start_run():
        # Training model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        pred_probs = model.predict_proba(X_test)[:, 1]

        # Hitung metrik performa
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, pred_probs)

        # Simpan dan log confusion matrix di root folder
        cm_path = "training_confusion_matrix.png"
        save_confusion_matrix(y_test, preds, cm_path)
        mlflow.log_artifact(cm_path)

        # Simpan metrik ke JSON dan TXT di root folder
        metric_info = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        }

        metric_json_path = "metric_info.json"
        with open(metric_json_path, "w") as f_json:
            json.dump(metric_info, f_json, indent=4)
        mlflow.log_artifact(metric_json_path)

        # Estimator report HTML di root folder
        estimator_html_path = "estimator.html"
        with open(estimator_html_path, "w") as f_html:
            f_html.write("<html><body><h1>Estimator Report</h1>")
            for k, v in metric_info.items():
                f_html.write(f"<p>{k}: {v}</p>")
            f_html.write("</body></html>")
        mlflow.log_artifact(estimator_html_path)

        # Simpan model.pkl secara manual di model/ dan log ke MLflow artifact
        model_pkl_path = os.path.join(artifact_dir, "model.pkl")
        joblib.dump(model, model_pkl_path)
        mlflow.log_artifact(model_pkl_path, artifact_path="model")

        # Siapkan input_example dan serving_input_example dalam JSON agar otomatis disimpan di model/
        input_example_df = X_train.head(5)
        input_example_path = os.path.join(artifact_dir, "input_example.json")
        input_example_df.to_json(input_example_path, orient='records', lines=False)
        
        serving_example_df = X_test.head(5)
        serving_example_path = os.path.join(artifact_dir, "serving_input_example.json")
        serving_example_df.to_json(serving_example_path, orient='records', lines=False)

        # Log file input_example.json dan serving_input_example.json secara manual ke MLflow artifact dengan artifact_path 'model' agar masuk ke folder model/
        mlflow.log_artifact(input_example_path, artifact_path="model")
        mlflow.log_artifact(serving_example_path, artifact_path="model")

        # Gunakan mlflow.sklearn.log_model untuk menyimpan model dengan conda.yaml, MLmodel, python_env.yaml otomatis dibuat di folder model/
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # folder model agar autosave conda.yaml dll di situ
            input_example=input_example_df
        )

    print("Training selesai. Semua artifacts tersimpan di lokal dan tercatat di MLflow sesuai struktur yang diminta.")

if __name__ == "__main__":
    main()