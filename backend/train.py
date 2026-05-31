import os
import subprocess
import mlflow
import shutil
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
# =========================
# CONFIG
# =========================
BASE_PRETRAIN = "yolov8n.pt"
LAST_MODEL = "model/last.pt"
DATA_YAML = "dataset.yaml"
VERSION_FILE = "version.txt"

#MLFLOW_URI = "http://mlflow-monitor:5000"
MLFLOW_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "camera_auto_train"

# =========================
# HYPERPARAMETERS (🔥 ADD)
# =========================
HYPERPARAMS = {
    "epochs": 1,
    "imgsz": 640,
    "batch": 16,
    "workers": 0,
    "optimizer": "auto",
    "lr0": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005
}

# =========================
# AUGMENTATION PARAMS (🔥 ADD)
# =========================
AUG_PARAMS = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "mosaic": 1.0,
    "mixup": 0.0
}

# =========================
# SETUP MLFLOW
# =========================
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.enable_system_metrics_logging()

os.environ["REPORT_TO"] = "none"

# =========================
# STRATEGY
# =========================
def get_strategy():
    if not os.path.exists("decision.txt"):
        return "first_time"
    return open("decision.txt").read().strip()

# =========================
# VERSION
# =========================
def get_next_version():
    if not os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "w") as f:
            f.write("1")
        return 1

    with open(VERSION_FILE) as f:
        v = int(f.read().strip())

    v += 1

    with open(VERSION_FILE, "w") as f:
        f.write(str(v))

    return v

# =========================
# TRAIN YOLO (CLI)
# =========================


def train_yolo(project_path, model_path):
    cmd = [
        "yolo", "detect", "train",
        f"model={model_path}",
        f"data={DATA_YAML}",
        f"epochs={HYPERPARAMS['epochs']}",
        f"imgsz={HYPERPARAMS['imgsz']}",
        f"batch={HYPERPARAMS['batch']}",
        f"workers={HYPERPARAMS['workers']}",
        f"project={project_path}",
        "name=run",
        "exist_ok=False",
        "amp=False" 
    ]

    subprocess.run(cmd, check=True)

    run_dir = os.path.join("runs/detect", project_path, "run")
    best_path = os.path.join(run_dir, "weights", "best.pt")

    if not os.path.exists(best_path):
        raise Exception(f"❌Không tìm thấy model: {best_path}")

    return best_path, run_dir

# =========================
# LOG METRICS
# =========================
def log_metrics(run_dir):
    results_path = os.path.join(run_dir, "results.csv")

    if not os.path.exists(results_path):
        print("⚠️ Không có results.csv")
        return

    df = pd.read_csv(results_path)

    for i, row in df.iterrows():
        #mlflow.log_metric("mAP50", row["metrics/mAP50(B)"], step=i)
        #mlflow.log_metric("precision", row["metrics/precision(B)"], step=i)
        #mlflow.log_metric("recall", row["metrics/recall(B)"], step=i)
        #mlflow.log_metric("mAP50-95", row["metrics/mAP50-95(B)"], step=i)

        # 🔥 thêm loss
        mlflow.log_metric("train_box_loss", row["train/box_loss"], step=i)
        mlflow.log_metric("train_cls_loss", row["train/cls_loss"], step=i)
        mlflow.log_metric("val_box_loss", row["val/box_loss"], step=i)
        mlflow.log_metric("val_cls_loss", row["val/cls_loss"], step=i)

    # 🔥 log final metric
    last = df.iloc[-1]
    mlflow.log_metric("final_mAP50", last["metrics/mAP50(B)"])
    mlflow.log_metric("final_mAP50_95", last["metrics/mAP50-95(B)"])
    mlflow.log_metric("final_precision", last["metrics/precision(B)"])
    mlflow.log_metric("final_recall", last["metrics/recall(B)"])

    print("-- Logged metrics")

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    strategy = get_strategy()
    base_model = LAST_MODEL
    print("-- huấn luyện model")
    with mlflow.start_run() as run:

        run_id = run.info.run_id
        with open("mlflow_run_id.txt", "w") as f:
           f.write(run_id)
        # =========================
        # VERSION
        # =========================
        version = f"v{get_next_version()}"
        project_path = os.path.join("train_history", version)

        mlflow.log_param("data_version", version)
        mlflow.log_param("strategy", strategy)
        mlflow.log_param("base_model", base_model)

        print(f"\n-- Training: {project_path}")

        # =========================
        # 🔥 LOG HYPERPARAMS
        # =========================
        #mlflow.log_params({k: str(v) for k, v in HYPERPARAMS.items()})
        #mlflow.log_params({k: str(v) for k, v in AUG_PARAMS.items()})

        # =========================
        # TRAIN
        # =========================
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
        os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME
        os.environ["MLFLOW_RUN_ID"] = mlflow.active_run().info.run_id
        os.environ["YOLO_MLFLOW_DIRECT_LOGGING"] = "True"

        model_path, run_dir = train_yolo(project_path, base_model)

        # =========================
        # LOG METRICS
        # =========================
        log_metrics(run_dir)

        print(f"-- Model trained: {model_path}")

        # =========================
        # 🔥 LOG REAL YOLO CONFIG
        # =========================
        #args_path = os.path.join(run_dir, "args.yaml")

        #if os.path.exists(args_path):
           # with open(args_path) as f:
            #    yolo_args = yaml.safe_load(f)
            #mlflow.log_params({k: str(v) for k, v in yolo_args.items()})

        # =========================
        # SAVE MODEL
        # =========================
        #os.makedirs("model", exist_ok=True)
        #shutil.copy(model_path, LAST_MODEL)

        # =========================
        # LOG ARTIFACTS
        # =========================
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifacts(run_dir, artifact_path="yolo_outputs")
        # =========================
        # REGISTER MODEL 
        # =========================
        model_uri = f"runs:/{run.info.run_id}/model/best.pt"

        client = MlflowClient()

        # tạo registered model nếu chưa có
        try:
           client.create_registered_model("camera_detection_model")
        except:
           pass

        # tạo version mới
        result = client.create_model_version(
            name="camera_detection_model",
            source=f"runs:/{run.info.run_id}/model/best.pt",
            run_id=run.info.run_id
        )
        print("-- Registered model version:", result.version)
        # =========================
        # SAVE CURRENT MODEL PATH
        # =========================
        with open("current_model.txt", "w") as f:
            f.write(model_path)

        print("\n=== TRAIN DONE ===")
        print("Saved:", model_path)