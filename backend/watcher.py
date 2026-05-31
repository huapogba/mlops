import os
import time
import subprocess
import json
import mlflow

DATA_PATH = "static/history"
CHECK_INTERVAL = 10
DRIFT_THRESHOLD = 0.3
TRAIN_THRESHOLD = 100   # đủ 1000 ảnh mới train
STATE_FILE = "watcher_state.json"

if os.path.exists(STATE_FILE):
    last_count = json.load(open(STATE_FILE)).get("last_count", 0)
else:
    last_count = 0
#last_count = 0

def count_files(path):
    total = 0

    valid_ext = (".jpg", ".jpeg", ".png")

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(valid_ext):
                total += 1

    return total


def check_drift():
    print("-- Running drift detection...")

    result = subprocess.run(
        ["python", "backend/drift.py"],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    try:
        with open("drift_report.json") as f:
            drift = json.load(f)

        max_drift = max(drift.values()) if drift else 0
        return max_drift

    except Exception as e:
        print("❌ Cannot read drift report:", e)
        return 1


print("-- Watcher started...")

while True:
    try:
        current_count = count_files(DATA_PATH)

        # số ảnh mới kể từ lần train trước
        new_files = current_count - last_count

        #print(f"-- Current images: {current_count}")
        print(f"-- New images: {new_files}")

        # ✅ chỉ train khi đủ 1000 ảnh mới
        if new_files >= TRAIN_THRESHOLD:

            print(f"-- Reached {TRAIN_THRESHOLD} new images")

            # STEP 1: prepare data
            print("Preparing data...")
            subprocess.run(
                ["python", "backend/prepare_data.py"],
                check=True
            )

            # STEP 2: drift detection
            drift_score = check_drift()
            print(f"-- Drift score: {drift_score}")

            # STEP 3: decide train
            if drift_score > DRIFT_THRESHOLD:
                print("-- Drift detected → chạy DVC pipeline")

                subprocess.run(
                    ["dvc", "repro", "-f"],
                    check=True
                )

                # ✅ update sau khi train thành công
                last_count = current_count
                with open(STATE_FILE, "w") as f:
                  json.dump({"last_count": last_count}, f)

            else:
                print("-- Drift thấp → skip training")
            
            #last_count = current_count
        else:
            remain = TRAIN_THRESHOLD - new_files
            print(f"-- Need {remain} more images to retrain")
            
        # sau khi xử lý train / skip xong
        #with open(STATE_FILE, "w") as f:
        #   json.dump({"last_count": current_count}, f)
    except Exception as e:
        print("❌ Error:", e)

    time.sleep(CHECK_INTERVAL)