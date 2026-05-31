"""from ultralytics import YOLO
import cv2
import numpy as np
import time

# ================= PROMETHEUS =================
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import Counter


# =========================================================
# METRICS FOR GRAFANA
# =========================================================

# Total inference requests
INFERENCE_REQUESTS = Counter(
    "inference_requests_total",
    "Total inference requests"
)

# YOLO inference latency
INFERENCE_TIME = Histogram(
    "inference_seconds",
    "YOLO inference latency"
)

# Current FPS
FPS_METRIC = Gauge(
    "camera_fps",
    "Current camera FPS"
)

# Current detected people
PEOPLE_METRIC = Gauge(
    "people_count",
    "Detected people count"
)

# Person bbox count
PERSON_BBOX_METRIC = Gauge(
    "person_bbox_count",
    "Detected person bbox count"
)

# Head bbox count
HEAD_BBOX_METRIC = Gauge(
    "head_bbox_count",
    "Detected head bbox count"
)

# Current tracked IDs
TRACKED_IDS_METRIC = Gauge(
    "tracked_people_count",
    "Tracked people count"
)

# Total unique people ever detected
TOTAL_UNIQUE_PEOPLE_METRIC = Gauge(
    "total_unique_people",
    "Total unique people detected"
)


# =========================================================
# YOLO DETECTOR
# =========================================================

class YOLODetector:

    def __init__(
        self,
        model_path,
        conf_threshold=0.4,
        img_size=832,
        #tracker="bytetrack.yaml"
        tracker="botsort.yaml"
    ):

        # ================= YOLO MODEL =================
        self.model = YOLO(model_path, task="detect")

        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.tracker = tracker

        # ================= CLASS NAMES =================
        self.class_names = self.model.names

        # ================= FPS =================
        self.prev_time = time.time()

        # ================= TRACKING =================
        self.total_person_ids = set()

        # ================= TARGET CLASSES =================
        self.target_classes = [
            k for k, v in self.class_names.items()
            if v.lower() in ["head", "person"]
        ]

        print("===================================")
        print("YOLO Detector Initialized")
        print("Target Classes:", self.target_classes)
        print("Tracker:", self.tracker)
        print("===================================")

    # =====================================================
    # PROCESS FRAME
    # =====================================================

    def process_frame(self, frame):

        # ================= REQUEST COUNT =================
        INFERENCE_REQUESTS.inc()

        # ================= START TIMER =================
        start_time = time.time()

        # ================= YOLO TRACKING =================
        results = self.model.track(
            source=frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            classes=self.target_classes,
            verbose=False,
            half=True,
            device=0,
            persist=True,
            tracker=self.tracker
        )

        # ================= END TIMER =================
        latency = time.time() - start_time

        # ================= UPDATE LATENCY =================
        INFERENCE_TIME.observe(latency)

        # ================= FPS =================
        current_time = time.time()

        fps = 1 / max(current_time - self.prev_time, 1e-6)

        self.prev_time = current_time

        FPS_METRIC.set(round(fps, 2))

        # ================= RESULT =================
        result = results[0]

        detection_data = []

        # Current tracked IDs in current frame
        current_person_ids = set()

        person_bbox_count = 0
        head_bbox_count = 0

        # =====================================================
        # DETECTION
        # =====================================================

        if result.boxes is not None and len(result.boxes) > 0:

            boxes = result.boxes.xyxy.cpu().numpy()

            clss = result.boxes.cls.cpu().numpy()

            confs = result.boxes.conf.cpu().numpy()

            ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else np.full(len(boxes), -1)
            )

            heads = []
            persons = []

            # =================================================
            # CLASSIFY OBJECTS
            # =================================================

            for i in range(len(boxes)):

                class_name = self.class_names.get(
                    int(clss[i]),
                    ""
                ).lower()

                box_info = {
                    "id": int(ids[i]),
                    "bbox": boxes[i],
                    "conf": float(confs[i]),
                    "used": False
                }

                if class_name == "head":
                    heads.append(box_info)

                elif class_name == "person":
                    persons.append(box_info)

            # =================================================
            # PROCESS PERSONS
            # =================================================

            for p in persons:

                px1, py1, px2, py2 = p["bbox"]

                person_id = p["id"]

                person_bbox_count += 1

                # =============================================
                # TRACKING IDS
                # =============================================

                if person_id != -1:

                    current_person_ids.add(person_id)

                    self.total_person_ids.add(person_id)

                # =============================================
                # CHECK HEAD INSIDE PERSON
                # =============================================

                for h in heads:

                    if h["used"]:
                        continue

                    hx1, hy1, hx2, hy2 = h["bbox"]

                    # Head center point
                    hcx = (hx1 + hx2) / 2
                    hcy = (hy1 + hy2) / 2

                    # Check if head inside person
                    if (
                        px1 <= hcx <= px2
                        and
                        py1 <= hcy <= py2
                    ):
                        h["used"] = True

                # =============================================
                # COLOR BY TRACK ID
                # =============================================

                color = (
                    (person_id * 37) % 255,
                    (person_id * 17) % 255,
                    (person_id * 29) % 255
                )

                # =============================================
                # DRAW PERSON BOX
                # =============================================

                cv2.rectangle(
                    frame,
                    (int(px1), int(py1)),
                    (int(px2), int(py2)),
                    color,
                    2
                )

                # =============================================
                # LABEL
                # =============================================

                label = (
                    f"ID {person_id} | "
                    #f"{p['conf']:.2f}"
                )

                # =============================================
                # DRAW LABEL
                # =============================================

                cv2.putText(
                    frame,
                    label,
                    (int(px1) + 180, int(py1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                # =============================================
                # SAVE DETECTION
                # =============================================

                detection_data.append({
                    "id": person_id,
                    "class": "person",
                    "bbox": [
                        int(px1),
                        int(py1),
                        int(px2),
                        int(py2)
                    ],
                    "conf": round(float(p["conf"]), 3)
                })

            # =================================================
            # PROCESS REMAINING HEADS
            # =================================================

            for h in heads:

                if not h["used"]:

                    hx1, hy1, hx2, hy2 = h["bbox"]

                    head_bbox_count += 1

                    # =========================================
                    # DRAW HEAD BOX
                    # =========================================

                    cv2.rectangle(
                        frame,
                        (int(hx1), int(hy1)),
                        (int(hx2), int(hy2)),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "HEAD",
                        (int(hx1), int(hy1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                    # =========================================
                    # SAVE DETECTION
                    # =========================================

                    detection_data.append({
                        "id": h["id"],
                        "class": "head",
                        "bbox": [
                            int(hx1),
                            int(hy1),
                            int(hx2),
                            int(hy2)
                        ],
                        "conf": round(float(h["conf"]), 3)
                    })

        # =====================================================
        # FINAL COUNTS
        # =====================================================

        current_people_count = len(current_person_ids)

        total_unique_people = len(self.total_person_ids)

        counts = {
            "person": current_people_count,
            "head": head_bbox_count,
            "person_bbox": person_bbox_count,
            "tracked_ids": current_people_count,
            "total_unique_people": total_unique_people,
            "fps": round(fps, 2),
            "latency_ms": round(latency * 1000, 2)
        }

        # =====================================================
        # UPDATE GRAFANA METRICS
        # =====================================================

        FPS_METRIC.set(round(fps, 2))

        PEOPLE_METRIC.set(current_people_count)

        PERSON_BBOX_METRIC.set(person_bbox_count)

        HEAD_BBOX_METRIC.set(head_bbox_count)

        TRACKED_IDS_METRIC.set(current_people_count)

        TOTAL_UNIQUE_PEOPLE_METRIC.set(total_unique_people)

        return frame, counts, detection_data"""


from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch
from threading import Lock

# ================= PROMETHEUS =================

from prometheus_client import (
    Gauge,
    Histogram,
    Counter
)

# =========================================================
# METRICS
# =========================================================

INFERENCE_REQUESTS = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["camera_id"]
)

INFERENCE_TIME = Histogram(
    "inference_seconds",
    "YOLO inference latency"
)

FPS_METRIC = Gauge(
    "camera_fps",
    "Current camera FPS",
    ["camera_id"]
)

INFERENCE_FPS = Gauge(
    "inference_fps",
    "YOLO inference FPS",
    ["camera_id"]
)

PEOPLE_METRIC = Gauge(
    "people_count",
    "Detected people count",
    ["camera_id"]
)

TRACKED_IDS_METRIC = Gauge(
    "tracked_people_count",
    "Tracked people count",
    ["camera_id"]
)

DETECTION_TOTAL = Counter(
    "detection_total",
    "Total detections",
    ["camera_id", "class_name"]
)

GPU_MEMORY_MB = Gauge(
    "gpu_memory_mb",
    "GPU memory usage MB"
)

# =========================================================
# YOLO DETECTOR
# =========================================================

class YOLODetector:

    def __init__(
        self,
        model_path,
        conf_threshold=0.4,
        img_size=832,
        tracker="backend/bytetrack_custom.yaml"
    ):

        print("===================================")
        print(f"Loading model: {model_path}")

        # LOAD MODEL ONLY ONCE
        self.model = YOLO(
            model_path,
            task="detect"
        )

        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.tracker = tracker

        self.class_names = self.model.names

        # detect classes
        self.target_classes = [
            k for k, v in self.class_names.items()
            if v.lower() in ["person", "head"]
        ]

        # lock chống race condition GPU
        self.lock = Lock()

        # FPS tracker per camera
        self.camera_fps = {}

        print("YOLO Detector Initialized")
        print("Target Classes:", self.target_classes)
        print("Tracker:", tracker)
        print("CUDA:", torch.cuda.is_available())
        print("===================================")

    # =====================================================
    # PROCESS FRAME
    # =====================================================

    def process_frame(
        self,
        frame,
        camera_id="0"
    ):

        # =================================================
        # REQUEST COUNT
        # =================================================

        INFERENCE_REQUESTS.labels(
            camera_id=camera_id
        ).inc()

        start_time = time.time()

        # =================================================
        # YOLO TRACK
        # =================================================

        try:

            with self.lock:

                results = self.model.track(
                    source=frame,

                    imgsz=self.img_size,

                    conf=self.conf_threshold,

                    classes=self.target_classes,

                    verbose=False,

                    # IMPORTANT
                    persist=True,

                    tracker=self.tracker,

                    device=0 if torch.cuda.is_available() else "cpu",

                    half=torch.cuda.is_available()
                )

        except Exception as e:

            print(f"[YOLO ERROR] {e}")

            return frame, {
                "person": 0,
                "fps": 0,
                "latency_ms": 0
            }, []

        # =================================================
        # PERFORMANCE
        # =================================================

        latency = time.time() - start_time

        inference_fps = 1 / max(latency, 1e-6)

        if torch.cuda.is_available():

            GPU_MEMORY_MB.set(
                torch.cuda.memory_allocated(0) / 1024 / 1024
            )

        INFERENCE_FPS.labels(
            camera_id=camera_id
        ).set(round(inference_fps, 2))

        INFERENCE_TIME.observe(latency)

        # =================================================
        # REAL CAMERA FPS
        # =================================================

        now = time.time()

        if camera_id not in self.camera_fps:

            self.camera_fps[camera_id] = now

        prev_time = self.camera_fps[camera_id]

        real_fps = 1 / max(now - prev_time, 1e-6)

        self.camera_fps[camera_id] = now

        # =================================================
        # RESULT
        # =================================================

        result = results[0]

        detection_data = []

        person_count = 0

        current_ids = set()

        # =================================================
        # DETECTION
        # =================================================

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            boxes = result.boxes.xyxy.cpu().numpy()

            clss = result.boxes.cls.cpu().numpy()

            confs = result.boxes.conf.cpu().numpy()

            # TRACK IDS
            ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else np.full(len(boxes), -1)
            )

            detected_persons = 0

            for i in range(len(boxes)):

                cls_name = self.class_names[
                    int(clss[i])
                ].lower()

                # only person
                if cls_name != "person":
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    boxes[i]
                )

                track_id = int(ids[i])

                # skip no-id object
                if track_id == -1:
                    continue

                person_count += 1
                detected_persons += 1

                current_ids.add(track_id)

                # ================= COLOR =================

                color = (
                    (track_id * 37) % 255,
                    (track_id * 17) % 255,
                    (track_id * 29) % 255
                )

                # ================= BOX =================

                """cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # ================= LABEL =================

                label = f"ID {track_id}"

                # top-left
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )"""

                # ================= JSON =================

                detection_data.append({
                    "id": track_id,
                    "class": "person",
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(float(confs[i]), 3)
                })

            # =================================================
            # PROMETHEUS DETECTION COUNT
            # =================================================

            if detected_persons > 0:

                DETECTION_TOTAL.labels(
                    camera_id=camera_id,
                    class_name="person"
                ).inc(detected_persons)

        # =================================================
        # METRICS
        # =================================================

        FPS_METRIC.labels(
            camera_id=camera_id
        ).set(round(real_fps, 2))

        PEOPLE_METRIC.labels(
            camera_id=camera_id
        ).set(person_count)

        TRACKED_IDS_METRIC.labels(
            camera_id=camera_id
        ).set(len(current_ids))

        # =================================================
        # COUNTS
        # =================================================

        counts = {
            "person": person_count,

            # camera fps
            "fps": round(real_fps, 2),

            # yolo fps
            "inference_fps": round(
                inference_fps,
                2
            ),

            "latency_ms": round(
                latency * 1000,
                2
            )
        }

        return frame, counts, detection_data