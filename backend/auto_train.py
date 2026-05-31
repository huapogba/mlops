
import os

MAX_ROUNDS = 10

for i in range(MAX_ROUNDS):
    print(f"\n🔁 Round {i+1}")

    # train
    os.system("dvc repro train")

    # evaluate + decide
    os.system("dvc repro evaluate")
    os.system("dvc repro decide")

    # đọc signal
    with open("train_signal.txt") as f:
        signal = f.read().strip()

    print("📡 Signal:", signal)

    # 👉 STOP nếu accept
    if signal == "STOP":
        print("✅ Model accepted → STOP TRAINING")
        break
    else:
        print("❌ Not accepted → continue training")