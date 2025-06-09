import os
import json
import pickle
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mlflow.models import infer_signature

# Inisialisasi mlflow
dagshub.init(repo_owner='daffaakifah', repo_name='Membangun_model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/daffaakifah/Membangun_model.mlflow/")
mlflow.set_experiment("Heart Disease Classification")

def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(path)
    plt.close()

def save_estimator_html(model, path):
    html_content = f"""
    <html>
    <head><title>Estimator Details</title></head>
    <body>
        <h2>Random Forest Classifier Parameters</h2>
        <ul>
            <li>n_estimators: {model.n_estimators}</li>
            <li>max_depth: {model.max_depth}</li>
            <li>random_state: {model.random_state}</li>
        </ul>
    </body>
    </html>
    """
    with open(path, "w") as f:
        f.write(html_content)

def main():
    df = pd.read_csv("Membangun_model/heart_preprocessing.csv")
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}

    artifact_model_dir = "model"
    estimator_html_file = "estimator.html"
    cm_file = "training_confusion_matrix.png"
    metric_file = "metric_info.json"

    if not os.path.exists(artifact_model_dir):
        os.makedirs(artifact_model_dir)

    for n in param_grid['n_estimators']:
        for d in param_grid['max_depth']:
            with mlflow.start_run():
                model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                print(f"Run dengan n_estimators={n}, max_depth={d}, accuracy={acc:.4f}")

                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", d)
                mlflow.log_metric("accuracy", acc)

                signature = infer_signature(X_train, model.predict(X_train))

                # Log model dengan signature dan input_example
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=signature, input_example=X_train.iloc[:5])

                # Simpan model.pkl di folder model (overwrite file lama, tanpa hapus folder)
                model_pkl_path = os.path.join(artifact_model_dir, "model.pkl")
                with open(model_pkl_path, "wb") as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(model_pkl_path, artifact_path="model")

                # Simpan estimator.html (tidak dihapus setelah ini)
                save_estimator_html(model, estimator_html_file)
                mlflow.log_artifact(estimator_html_file)

                # Simpan confusion matrix image (tidak dihapus setelah ini)
                save_confusion_matrix(y_test, preds, cm_file)
                mlflow.log_artifact(cm_file)

                # Simpan metric info json (tidak dihapus setelah ini)
                with open(metric_file, "w") as f:
                    json.dump({"accuracy": acc}, f)
                mlflow.log_artifact(metric_file)

                print(f"Run untuk n_estimators={n}, max_depth={d} selesai dan semua artifact tersimpan.")

if __name__ == "__main__":
    main()