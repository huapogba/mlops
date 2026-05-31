import os
import json
import cv2
import shutil
import random
from sqlalchemy import create_engine, text

# =========================
# CONFIG
# =========================
BASE_IMAGE_DIR = "static/history"
#DATABASE_URL = "mysql+pymysql://root:tan150823@mysql-smart-camera:3306/smart_camera"
DATABASE_URL = "mysql+pymysql://root:tan150823@mysql:3306/smart_camera"
engine = create_engine(DATABASE_URL)

STATE_FILE = "data/version_state.json"
VERSION_ROOT = "data/versions"
LATEST_PATH = os.path.join(VERSION_ROOT, "latest")

os.makedirs(VERSION_ROOT, exist_ok=True)

# =========================
# VERSION
# =========================
def get_next_version():
    existing = [
        d for d in os.listdir(VERSION_ROOT)
        if d.startswith("v") and d[1:].isdigit()
    ]

    if not existing:
        return 1

    return max(int(v[1:]) for v in existing) + 1


# =========================
# LOAD STATE
# =========================
last_version = 0
if os.path.exists(STATE_FILE):
    try:
        last_version = json.load(open(STATE_FILE)).get("last_version", 0)
    except:
        pass

VERSION_NUM = max(last_version, get_next_version())
VERSION = f"v{VERSION_NUM}"
DATASET_ROOT = os.path.join(VERSION_ROOT, VERSION)

# tránh overwrite
if os.path.exists(DATASET_ROOT):
    VERSION_NUM = get_next_version()
    VERSION = f"v{VERSION_NUM}"
    DATASET_ROOT = os.path.join(VERSION_ROOT, VERSION)

print(f"-- Creating dataset version: {VERSION}")

# =========================
# CREATE STRUCTURE (train/val)
# =========================
IMG_TRAIN = os.path.join(DATASET_ROOT, "images/train")
IMG_VAL   = os.path.join(DATASET_ROOT, "images/val")

LBL_TRAIN = os.path.join(DATASET_ROOT, "labels/train")
LBL_VAL   = os.path.join(DATASET_ROOT, "labels/val")

for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(d, exist_ok=True)

# =========================
# CLASS MAP
# =========================
CLASS_MAP = {"person": 0, "head": 1}

# =========================
# LOAD DATA FROM DB
# =========================
with engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT c.image_path, d.class_name, d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2
        FROM detections d
        JOIN camera_logs c ON d.log_id = c.id
        WHERE c.image_path IS NOT NULL
    """)).fetchall()

data = {}

for r in rows:
    img_name = os.path.basename(r.image_path)
    img_src = os.path.join(BASE_IMAGE_DIR, img_name)

    if not os.path.exists(img_src):
        continue

    data.setdefault(img_name, {"src": img_src, "objects": []})
    data[img_name]["objects"].append(r)

if not data:
    print("❌ No valid data")
    exit(1)

# =========================
# SPLIT 8:2
# =========================
random.seed(42)

all_images = list(data.keys())
random.shuffle(all_images)

split_idx = int(len(all_images) * 0.8)

train_set = set(all_images[:split_idx])
val_set   = set(all_images[split_idx:])

print(f"-- Train: {len(train_set)} | Val: {len(val_set)}")

# =========================
# BUILD DATASET
# =========================
valid_images = 0

for img_name, item in data.items():

    img_src = item["src"]
    img = cv2.imread(img_src)
    if img is None:
        continue

    h, w = img.shape[:2]

    # chọn folder
    if img_name in train_set:
        img_dir = IMG_TRAIN
        lbl_dir = LBL_TRAIN
    else:
        img_dir = IMG_VAL
        lbl_dir = LBL_VAL

    label_path = os.path.join(
        lbl_dir,
        os.path.splitext(img_name)[0] + ".txt"
    )

    has_label = False

    with open(label_path, "w") as f:
        for obj in item["objects"]:

            if obj.class_name not in CLASS_MAP:
                continue

            x1, y1, x2, y2 = obj.bbox_x1, obj.bbox_y1, obj.bbox_x2, obj.bbox_y2
            if x2 <= x1 or y2 <= y1:
                continue

            cid = CLASS_MAP[obj.class_name]

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            f.write(f"{cid} {xc} {yc} {bw} {bh}\n")
            has_label = True

    if has_label:
        shutil.copy(img_src, os.path.join(img_dir, img_name))
        valid_images += 1
    else:
        os.remove(label_path)

# =========================
# FINAL CHECK
# =========================
if valid_images == 0:
    print("❌ No valid images → delete version folder")
    shutil.rmtree(DATASET_ROOT)
    exit(1)

# =========================
# SAVE STATE
# =========================
with open(STATE_FILE, "w") as f:
    json.dump({"last_version": VERSION_NUM}, f, indent=2)

# =========================
# UPDATE LATEST
# =========================
if os.path.exists(LATEST_PATH):
    shutil.rmtree(LATEST_PATH)

shutil.copytree(DATASET_ROOT, LATEST_PATH)

# =========================
# OUTPUT
# =========================
print("======================================")
print("-- DATASET VERSION CREATED")
print(f"-- Version: {VERSION}")
print(f"-- Path: {DATASET_ROOT}")
print(f"-- Images: {valid_images}")
print("-- Latest updated (train/val ready)")
print("======================================")