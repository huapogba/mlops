import os
import cv2
import time
import uuid
import base64
import asyncio
import logging
import requests
import numpy as np
import pytz

from datetime import datetime
from threading import Lock

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Depends
)

from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response
)

from fastapi.staticfiles import StaticFiles

# ================= SQLALCHEMY =================

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Float,
    ForeignKey
)

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import (
    sessionmaker,
    Session,
    relationship
)

# ================= PROMETHEUS =================

from prometheus_client import generate_latest

# ================= AI DETECTOR =================

from backend.yolo_model import YOLODetector


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI()

# =========================================================
# DATABASE
# =========================================================

"""DATABASE_URL = (
    "mysql+pymysql://root:tan150823@mysql-smart-camera:3306/smart_camera"
)"""

DATABASE_URL = (
    "mysql+pymysql://root:tan150823@mysql:3306/smart_camera"
)
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

# =========================================================
# MODELS
# =========================================================

class Camera(Base):

    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(100), nullable=False)

    rtsp_url = Column(String(255))

    status = Column(Integer, default=1)

    logs = relationship(
        "CameraLog",
        back_populates="camera"
    )


class CameraLog(Base):

    __tablename__ = "camera_logs"

    id = Column(Integer, primary_key=True, index=True)

    camera_id = Column(
        Integer,
        ForeignKey("cameras.id")
    )

    person_count = Column(Integer, default=0)

    head_count = Column(Integer, default=0)

    image_path = Column(String(255))

    created_at = Column(DateTime, default=None)

    camera = relationship(
        "Camera",
        back_populates="logs"
    )

    detections = relationship(
        "Detection",
        back_populates="log",
        cascade="all, delete"
    )


class Detection(Base):

    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)

    log_id = Column(
        Integer,
        ForeignKey(
            "camera_logs.id",
            ondelete="CASCADE"
        )
    )

    class_name = Column(String(50))

    bbox_x1 = Column(Float)

    bbox_y1 = Column(Float)

    bbox_x2 = Column(Float)

    bbox_y2 = Column(Float)

    confidence = Column(Float)

    track_id = Column(Integer, nullable=True)

    log = relationship(
        "CameraLog",
        back_populates="detections"
    )


Base.metadata.create_all(bind=engine)

# =========================================================
# DB SESSION
# =========================================================

def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# =========================================================
# STORAGE
# =========================================================

CAMERA_STREAMS = {}

CAMERA_ACTIVE = {}

CAMERA_FAIL_COUNT = {}

CAMERA_NAMES = {}

CAMERA_STREAMS_LOCK = Lock()

# =========================================================
# HISTORY
# =========================================================

HISTORY_DIR = "static/history"

os.makedirs(HISTORY_DIR, exist_ok=True)

# =========================================================
# STATIC
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)

app.mount(
    "/history",
    StaticFiles(directory=HISTORY_DIR),
    name="history"
)

# =========================================================
# YOLO
# =========================================================

detector = YOLODetector(
    "model/best.engine"
)

# =========================================================
# TELEGRAM
# =========================================================

BOT_TOKEN = "8237330723:AAG0GG1hRjA90rBqYEREHPbR4k60CWDdIVg"

CHAT_ID = "5947187813"

last_alert_time = {}

telegram_queue = None

# =========================================================
# TELEGRAM WORKER
# =========================================================

async def init_telegram_queue():

    global telegram_queue

    telegram_queue = asyncio.Queue(maxsize=20)

    asyncio.create_task(
        telegram_worker()
    )


async def telegram_worker():

    while True:

        try:

            frame, detections, camera_id = await asyncio.wait_for(
                telegram_queue.get(),
                timeout=300
            )

            await asyncio.to_thread(
                send_telegram,
                frame,
                detections,
                camera_id
            )

            await asyncio.sleep(0.2)

        except asyncio.TimeoutError:
            continue

        except Exception as e:

            logger.error(
                f"Telegram worker error: {e}"
            )

# =========================================================
# HELPERS
# =========================================================

def frame_to_bytes(frame):

    _, buffer = cv2.imencode(".jpg", frame)

    return buffer.tobytes()


def draw_bboxes_on_image(
    image_bytes,
    detections_list,
    canvas_width=None,
    canvas_height=None
):
    nparr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]

    # nếu không truyền canvas thì dùng size gốc
    canvas_width = canvas_width or w
    canvas_height = canvas_height or h

    scale_x = canvas_width / w
    scale_y = canvas_height / h

    for det in detections_list:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox

        # scale giống JS
        draw_x = int(x1 * scale_x)
        draw_y = int(y1 * scale_y)
        draw_w = int((x2 - x1) * scale_x)
        draw_h = int((y2 - y1) * scale_y)

        track_id = det.get("id", -1)
        class_name = det.get("class", "obj")

        # ===== BOX =====
        color = (0, 255, 0)  # giống "#00FF00"
        cv2.rectangle(
            img,
            (draw_x, draw_y),
            (draw_x + draw_w, draw_y + draw_h),
            color,
            2
        )

        # ===== LABEL TEXT =====
        label = f"{class_name} ID:{track_id}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # ===== LABEL BACKGROUND =====
        cv2.rectangle(
            img,
            (draw_x, draw_y - text_h - 6),
            (draw_x + text_w + 6, draw_y),
            color,
            -1
        )

        # ===== TEXT =====
        cv2.putText(
            img,
            label,
            (draw_x + 3, draw_y - 4),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    _, encoded_img = cv2.imencode(".jpg", img)
    
    if encoded_img is None or len(encoded_img) == 0:
       return None

    return encoded_img.tobytes()


def send_telegram(
    frame,
    detections,
    camera_id
):

    cam_id = str(camera_id)

    camera_name = CAMERA_NAMES.get(
        cam_id,
        f"Camera {cam_id}"
    )

    image_bytes = frame_to_bytes(frame)

    img_with_bb = draw_bboxes_on_image(
        image_bytes,
        detections
    )

    if img_with_bb is None:
        return

    temp_file = f"temp_{cam_id}.jpg"

    try:

        with open(temp_file, "wb") as f:
            f.write(img_with_bb)

        url = (
            f"https://api.telegram.org/bot"
            f"{BOT_TOKEN}/sendPhoto"
        )

        with open(temp_file, "rb") as photo:

            requests.post(
                url,
                files={
                    "photo": photo
                },
                data={
                    "chat_id": CHAT_ID,
                    "caption":
                    f"🚨 {camera_name} phát hiện người"
                },
                timeout=5
            )

        logger.info(
            f"Telegram sent: {camera_name}"
        )

    except Exception as e:

        logger.error(
            f"Telegram error: {e}"
        )

    finally:

        if os.path.exists(temp_file):
            os.remove(temp_file)

# =========================================================
# RTSP
# =========================================================

def create_rtsp(rtsp):

    cap = cv2.VideoCapture(
        rtsp,
        cv2.CAP_FFMPEG
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    for _ in range(10):

        if cap.isOpened():
            return cap

        time.sleep(1)

        cap = cv2.VideoCapture(
            rtsp,
            cv2.CAP_FFMPEG
        )

    return None

# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
async def startup_event():

    logger.info("Starting server...")

    await init_telegram_queue()

    load_cameras_from_db()

    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown_event():

    logger.info("Shutdown server...")

    with CAMERA_STREAMS_LOCK:

        for cam_id in CAMERA_ACTIVE:
            CAMERA_ACTIVE[cam_id] = False

    await asyncio.sleep(1)

    logger.info("Shutdown complete")

# =========================================================
# LOAD CAMERAS
# =========================================================

def load_cameras_from_db():

    db = SessionLocal()

    try:

        cameras = db.query(Camera).filter(
            Camera.status == 1
        ).all()

        with CAMERA_STREAMS_LOCK:

            for cam in cameras:

                cam_id = str(cam.id)

                CAMERA_STREAMS[cam_id] = (
                    cam.rtsp_url
                )

                CAMERA_ACTIVE[cam_id] = True

                CAMERA_FAIL_COUNT[cam_id] = 0

                CAMERA_NAMES[cam_id] = cam.name

        logger.info(
            f"Loaded {len(cameras)} cameras"
        )

    except Exception as e:

        logger.error(
            f"Load camera error: {e}"
        )

    finally:
        db.close()

# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "total_cameras": len(CAMERA_STREAMS)
    }

# =========================================================
# METRICS
# =========================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )

# =========================================================
# HOME
# =========================================================

@app.get("/")
async def home():

    with open(
        "frontend/main.html",
        encoding="utf-8"
    ) as f:

        return HTMLResponse(
            f.read()
        )

# =========================================================
# GET CAMERAS
# =========================================================

@app.get("/cameras")
def get_cameras(
    db: Session = Depends(get_db)
):

    cams = db.query(Camera).filter(
        Camera.status == 1
    ).all()

    return [

        {
            "id": cam.id,
            "name": cam.name,
            "rtsp": cam.rtsp_url,
            "status": cam.status
        }

        for cam in cams
    ]

# =========================================================
# ADD CAMERA
# =========================================================

@app.post("/camera/add")
async def add_camera(
    data: dict,
    db: Session = Depends(get_db)
):

    name = data.get("name")

    rtsp = data.get("rtsp")

    if not name or not rtsp:

        return JSONResponse(
            {
                "error": "Thiếu dữ liệu"
            },
            status_code=400
        )

    cap = cv2.VideoCapture(
        rtsp,
        cv2.CAP_FFMPEG
    )

    if not cap.isOpened():

        return JSONResponse(
            {
                "error": "Không kết nối được RTSP"
            },
            status_code=400
        )

    ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:

        return JSONResponse(
            {
                "error": "RTSP không hợp lệ"
            },
            status_code=400
        )

    try:

        new_cam = Camera(
            name=name,
            rtsp_url=rtsp,
            status=1
        )

        db.add(new_cam)

        db.commit()

        db.refresh(new_cam)

        cam_id = str(new_cam.id)

        with CAMERA_STREAMS_LOCK:

            CAMERA_STREAMS[cam_id] = rtsp

            CAMERA_ACTIVE[cam_id] = True

            CAMERA_FAIL_COUNT[cam_id] = 0

            CAMERA_NAMES[cam_id] = name

        return {
            "status": "success",
            "camera_id": new_cam.id
        }

    except Exception as e:

        db.rollback()

        return JSONResponse(
            {
                "error": str(e)
            },
            status_code=500
        )

# =========================================================
# IMAGE DETECT
# =========================================================

@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...)
):

    contents = await file.read()

    np_arr = np.frombuffer(
        contents,
        np.uint8
    )

    img = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    frame, counts, detections = (
        detector.process_frame(img)
    )

    _, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    base64_str = base64.b64encode(
        buffer
    ).decode("utf-8")

    return {
        "image":
        f"data:image/jpeg;base64,{base64_str}",

        "count": counts["person"],

        "fps": counts["fps"],

        "detections": detections
    }

@app.get("/camera/history")
async def get_history(
    db: Session = Depends(get_db)
):

    results = db.query(
        CameraLog,
        Camera.name
    ).join(
        Camera,
        CameraLog.camera_id == Camera.id
    ).order_by(
        CameraLog.id.desc()
    ).limit(50).all()

    output = []

    for log, cam_name in results:

        dets = db.query(
            Detection
        ).filter(
            Detection.log_id == log.id
        ).all()

        det_list = []

        for d in dets:

            det_list.append({

                "class": d.class_name,

                "bbox": [
                    d.bbox_x1,
                    d.bbox_y1,
                    d.bbox_x2,
                    d.bbox_y2
                ],

                "conf": d.confidence,

                "id": d.track_id
            })

        output.append({

            "id": log.id,

            "camera_name": cam_name,

            "image_url": log.image_path,

            "count": log.person_count,

            "detections": det_list,

            "created_at": log.created_at.strftime(
                "%H:%M:%S %d/%m/%Y"
            )
        })

    return output

@app.post("/camera/save_log")
async def save_log():

    return {
        "status": "ok"
    }
# =========================================================
# SIMPLE VIDEO WS
# =========================================================

@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):

    await websocket.accept()

    try:

        while True:

            data = await websocket.receive_json()

            if "frame" not in data:
                continue

            img_data = data["frame"].split(",")[1]

            img_bytes = base64.b64decode(
                img_data
            )

            np_arr = np.frombuffer(
                img_bytes,
                np.uint8
            )

            frame = cv2.imdecode(
                np_arr,
                cv2.IMREAD_COLOR
            )

            frame, counts, detections = await asyncio.to_thread(
                detector.process_frame,
                frame,
                "0"
            )

            _, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            jpg = base64.b64encode(
                buffer
            ).decode()

            await websocket.send_json({

                "frame": jpg,

                "count": counts["person"],

                "fps": counts["fps"],

                "detections": detections
            })

    except WebSocketDisconnect:

        logger.info(
            "Video WS disconnected"
        )

# =========================================================
# DELETE CAMERA
# =========================================================

@app.delete("/camera/{camera_id}")
async def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db)
):

    try:

        cam = db.query(Camera).filter(
            Camera.id == camera_id
        ).first()

        if not cam:

            return JSONResponse(
                {
                    "error": "Camera not found"
                },
                status_code=404
            )

        # update status
        cam.status = 0

        db.commit()

        cam_id = str(camera_id)

        # stop websocket stream
        with CAMERA_STREAMS_LOCK:

            CAMERA_ACTIVE[cam_id] = False

            if cam_id in CAMERA_STREAMS:
                del CAMERA_STREAMS[cam_id]

            if cam_id in CAMERA_FAIL_COUNT:
                del CAMERA_FAIL_COUNT[cam_id]

            if cam_id in CAMERA_NAMES:
                del CAMERA_NAMES[cam_id]

        logger.info(
            f"🗑️ Camera deleted: {camera_id}"
        )

        return {
            "status": "success"
        }

    except Exception as e:

        db.rollback()

        logger.error(
            f"❌ Delete camera error: {e}"
        )

        return JSONResponse(
            {
                "error": str(e)
            },
            status_code=500
        )
# =========================================================
# CAMERA WS
# =========================================================

@app.websocket("/ws/camera/{camera_id}")
async def camera_ws(
    websocket: WebSocket,
    camera_id: str
):

    await websocket.accept()

    logger.info(
        f"📡 Camera connected: {camera_id}"
    )

    # =====================================================
    # GET RTSP
    # =====================================================

    with CAMERA_STREAMS_LOCK:

        rtsp = CAMERA_STREAMS.get(camera_id)

    if not rtsp:

        logger.error(
            f"❌ Camera not found: {camera_id}"
        )

        await websocket.close()

        return

    # =====================================================
    # OPEN RTSP
    # =====================================================

    cap = create_rtsp(rtsp)

    if cap is None:

        logger.error(
            f"❌ Cannot open RTSP: {camera_id}"
        )

        await websocket.close()

        return

    # =====================================================
    # SETTINGS
    # =====================================================

    frame_skip = 2

    frame_count = 0

    reconnect_count = 0

    # =====================================================
    # LOOP
    # =====================================================

    try:

        while True:

            # =================================================
            # CAMERA ACTIVE CHECK
            # =================================================

            with CAMERA_STREAMS_LOCK:

                is_active = CAMERA_ACTIVE.get(
                    camera_id,
                    True
                )

            if not is_active:

                await asyncio.sleep(0.5)

                continue

            # =================================================
            # REDUCE RTSP DELAY
            # =================================================

            try:

                for _ in range(2):
                    cap.grab()

                ret, frame = cap.retrieve()

            except Exception as e:

                logger.error(
                    f"❌ RTSP read exception {camera_id}: {e}"
                )

                ret = False
                frame = None

            # =================================================
            # RECONNECT RTSP
            # =================================================

            if not ret or frame is None:

                reconnect_count += 1

                logger.warning(
                    f"⚠️ Camera read fail {camera_id} | reconnect={reconnect_count}"
                )

                try:
                    cap.release()
                except:
                    pass

                await asyncio.sleep(1)

                cap = create_rtsp(rtsp)

                if cap is None:

                    logger.error(
                        f"❌ Reconnect failed {camera_id}"
                    )

                    await asyncio.sleep(2)

                continue

            reconnect_count = 0

            # =================================================
            # FRAME SKIP
            # =================================================

            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            # =================================================
            # YOLO DETECT
            # =================================================

            try:

                processed_frame, counts, detections = await asyncio.to_thread(
                    detector.process_frame,
                    frame.copy(),
                    camera_id
                )

            except Exception as e:

                logger.error(
                    f"❌ YOLO error {camera_id}: {e}"
                )

                continue

            # =================================================
            # PERSON DETECTED
            # =================================================

            person_count = counts.get(
                "person",
                0
            )

            # =================================================
            # SAVE IMAGE + TELEGRAM
            # =================================================

            try:

                now = time.time()

                last = last_alert_time.get(
                    camera_id,
                    0
                )

                # cooldown 30s
                if person_count > 0 and (now - last > 30):

                    # =========================================
                    # SAVE IMAGE
                    # =========================================

                    tz = pytz.timezone(
                        "Asia/Ho_Chi_Minh"
                    )

                    timestamp = datetime.now(tz).strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    filename = (
                        f"cam_{camera_id}_"
                        f"{timestamp}_"
                        f"{uuid.uuid4().hex[:6]}.jpg"
                    )

                    filepath = os.path.join(
                        HISTORY_DIR,
                        filename
                    )

                    save_ok = cv2.imwrite(
                        filepath,
                        processed_frame
                    )

                    if save_ok:

                        logger.info(
                            f"✅ Saved image: {filename}"
                        )

                        # =====================================
                        # SAVE DATABASE
                        # =====================================

                        db = SessionLocal()

                        try:

                            new_log = CameraLog(
                                camera_id=int(camera_id),
                                person_count=person_count,
                                head_count=0,
                                image_path=f"/history/{filename}",
                                created_at=datetime.now()
                            )

                            db.add(new_log)

                            db.flush()

                            # SAVE DETECTIONS
                            for det in detections:

                                bbox = det.get(
                                    "bbox",
                                    [0, 0, 0, 0]
                                )

                                x1, y1, x2, y2 = map(
                                    float,
                                    bbox
                                )

                                new_det = Detection(
                                    log_id=new_log.id,
                                    class_name=det.get("class"),
                                    bbox_x1=x1,
                                    bbox_y1=y1,
                                    bbox_x2=x2,
                                    bbox_y2=y2,
                                    confidence=det.get("conf"),
                                    track_id=det.get("id")
                                )

                                db.add(new_det)

                            db.commit()

                            logger.info(
                                f"✅ DB log saved camera={camera_id}"
                            )

                        except Exception as e:

                            db.rollback()

                            logger.error(
                                f"❌ DB save error: {e}"
                            )

                        finally:

                            db.close()

                    # =========================================
                    # TELEGRAM ALERT
                    # =========================================

                    if telegram_queue is not None:

                        if not telegram_queue.full():

                            await telegram_queue.put((
                                processed_frame.copy(),
                                detections,
                                camera_id
                            ))

                            logger.info(
                                f"✅ Telegram queued camera={camera_id}"
                            )

                    # update cooldown
                    last_alert_time[camera_id] = now

            except Exception as e:

                logger.error(
                    f"❌ Alert error {camera_id}: {e}"
                )

            # =================================================
            # ENCODE JPEG
            # =================================================

            try:

                success, buffer = cv2.imencode(
                    ".jpg",
                    processed_frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        60
                    ]
                )

                if not success:
                    continue

                jpg_as_text = base64.b64encode(
                    buffer.tobytes()
                ).decode("ascii")

            except Exception as e:

                logger.error(
                    f"❌ JPEG encode error {camera_id}: {e}"
                )

                continue

            # =================================================
            # SEND FRONTEND
            # =================================================

            try:

                await websocket.send_json({

                    "frame": jpg_as_text,

                    "count": person_count,

                    "fps": counts.get(
                        "fps",
                        0
                    ),

                    "latency_ms": counts.get(
                        "latency_ms",
                        0
                    ),

                    "detections": detections
                })

            except WebSocketDisconnect:

                logger.info(
                    f"🔌 Client disconnected {camera_id}"
                )

                break

            except Exception as e:

                logger.error(
                    f"❌ WS send error {camera_id}: {e}"
                )

                break

            # =================================================
            # SMALL DELAY
            # =================================================

            await asyncio.sleep(0.01)

    # =====================================================
    # FINALLY
    # =====================================================

    finally:

        try:

            if cap:
                cap.release()

        except:
            pass

        logger.info(
            f"🔴 Camera closed {camera_id}"
        )