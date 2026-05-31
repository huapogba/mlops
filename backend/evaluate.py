import os
import json
import shutil
import mlflow
from ultralytics import YOLO

# =========================
# CONFIG MLFLOW
# =========================
#MLFLOW_URI = "http://mlflow-monitor:5000"
MLFLOW_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "camera_auto_train"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# =========================
# MAIN EVALUATE
# =========================
def evaluate():
    if not os.path.exists("current_model.txt"):
        raise Exception("❌ Không có model mới")

    with open("current_model.txt") as f:
        model_path = f.read().strip()

    print(f"Evaluating: {model_path}")

    model = YOLO(model_path)
    metrics = model.val(data="dataset.yaml", workers=0)

    result = {
        "version": os.path.basename(model_path),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    # =========================
    # 🔥 LOAD HISTORY
    # =========================
    history = []

    if os.path.exists("metrics.json"):
        try:
            with open("metrics.json") as f:
                history = json.load(f)

            if isinstance(history, dict):
                history = [history]

        except:
            history = []

    # =========================
    # 🔥 APPEND
    # =========================
    history.append(result)

    # =========================
    # 🔥 SAVE BACK
    # =========================
    with open("metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    shutil.copy("drift_report.json", "drift_baseline.json")

    #print("-- Metrics:", result)
    print(f"-- Total history: {len(history)} runs")

    # =========================
    # 🚀 MLFLOW LOGGING (🔥 ADD)
    # =========================
    run_id = None

    if os.path.exists("mlflow_run_id.txt"):
       with open("mlflow_run_id.txt") as f:
         run_id = f.read().strip()
    with mlflow.start_run(run_id=run_id):

        
        # metrics (validation)
        mlflow.log_metric("val_precision", result["precision"])
        mlflow.log_metric("val_recall", result["recall"])
        mlflow.log_metric("val_mAP50", result["map50"])
        mlflow.log_metric("val_mAP50_95", result["map50_95"])

        # log model file
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="model")


        print("-- Logged evaluation to MLflow")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    evaluate()