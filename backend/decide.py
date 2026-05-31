import json
import yaml
import shutil
import os

CURRENT_MODEL_FILE = "current_model.txt"
LAST_MODEL = "model/last.pt"

def decide():
    with open("metrics.json") as f:
        metrics = json.load(f)

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    if not metrics:
        raise ValueError("metrics.json is empty")

    # =========================
    # lấy record mới nhất
    # =========================
    latest = metrics[-1] if isinstance(metrics, list) else metrics

    # =========================
    # tính score
    # =========================
    score = (
        latest.get("map50", 0) * 0.5 +
        latest.get("precision", 0) * 0.25 +
        latest.get("recall", 0) * 0.25
    )

    bad = params["thresholds"]["score_bad"]
    ok = params["thresholds"]["score_ok"]

    if score < bad:
        strategy = "retrain"
    elif score < ok:
        strategy = "fine_tune"
    else:
        strategy = "accept"

    # =========================
    # save decision
    # =========================
    with open("decision.txt", "w") as f:
        f.write(strategy)

    print(f"-- Decision: {strategy} | score={score:.3f}")

    # =========================
    # 🔥 đọc model path từ file
    # =========================
    if not os.path.exists(CURRENT_MODEL_FILE):
        raise FileNotFoundError("current_model.txt not found")

    with open(CURRENT_MODEL_FILE) as f:
        best_model = f.read().strip()

    # =========================
    # promote
    # =========================
    
    BEST_SCORE_FILE = "best_score.json"

    def load_best_score():
      if os.path.exists(BEST_SCORE_FILE):
        with open(BEST_SCORE_FILE) as f:
            return json.load(f).get("score", 0)
      return 0

    def save_best_score(score):
      with open(BEST_SCORE_FILE, "w") as f:
        json.dump({"score": score}, f)

    #=========================
    # PROMOTE LOGIC
    #=========================
    best_score = load_best_score()

    if strategy == "accept":
       if not os.path.exists(best_model):
         print("❌ Model path invalid:", best_model)
       else:
         if score > best_score:
            os.makedirs("model", exist_ok=True)
            shutil.copy(best_model, LAST_MODEL)

            save_best_score(score)

            print(f"✅ Promoted (better): {best_model} → {LAST_MODEL}")
            print(f"📈 Score improved: {best_score:.3f} → {score:.3f}")
         else:
            print(f"⏭️ Model không tốt hơn ({score:.3f} <= {best_score:.3f}) → giữ model cũ")
    else:
      print("keep old model")

   

if __name__ == "__main__":
    decide()