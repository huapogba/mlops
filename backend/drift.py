import os
import json

# =========================
# CONFIG
# =========================
LABEL_DIR = "data/versions/latest/labels/train"
BASELINE_FILE = "drift_baseline.json"
OUTPUT_FILE = "drift_report.json"

# =========================
# LOAD LABELS
# =========================
def load_labels(label_dir):
    class_counts = {}

    if not os.path.exists(label_dir):
        print("⚠️ Label dir not found:", label_dir)
        return class_counts

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(label_dir, file)

        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue

                cls = int(parts[0])
                class_counts[cls] = class_counts.get(cls, 0) + 1

    return class_counts

# =========================
# COMPUTE DRIFT
# =========================
def compute_drift(current, baseline):
    drift_score = {}

    all_classes = set(current.keys()) | set(baseline.keys())

    for cls in all_classes:
        c = current.get(cls, 0)
        b = baseline.get(cls, 0)

        # 🔥 safe division + bounded drift
        drift = abs(c - b) / (b + 1e-6)
        drift_score[str(cls)] = 1.0 
        #min(drift, 1.0)

    return drift_score

# =========================
# MAIN
# =========================
def main():
    # 📊 current distribution
    current_dist = load_labels(LABEL_DIR)

    # =========================
    # BASELINE HANDLING
    # =========================
    if not os.path.exists(BASELINE_FILE):
        # create baseline
        with open(BASELINE_FILE, "w") as f:
            json.dump(current_dist, f, indent=2)

        print("📌 Baseline created")

        # no drift in first run
        drift = {str(k): 0.0 for k in current_dist}

    else:
        # load baseline
        with open(BASELINE_FILE) as f:
            baseline_dist = json.load(f)

        # 🔥 FIX: convert key type (JSON → string)
        baseline_dist = {int(k): v for k, v in baseline_dist.items()}

        drift = compute_drift(current_dist, baseline_dist)

    # =========================
    # SAVE REPORT
    # =========================
    with open(OUTPUT_FILE, "w") as f:
        json.dump(drift, f, indent=2)

    # =========================
    # MAX DRIFT
    # =========================
    max_drift = max(drift.values()) if drift else 0.0

    print("-- Drift per class:", drift)
    print(f"-- Max drift: {max_drift}")

    return max_drift

# =========================
if __name__ == "__main__":
    main()