// =======================
// TAB SWITCH
// =======================
function showTab(tabId, btn) {
    // 1. Chuyển đổi trạng thái Active của Tab
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".sidebar button").forEach(b => b.classList.remove("active"));
    
    document.getElementById(tabId).classList.add("active");
    btn.classList.add("active");

    // 2. Logic bổ sung khi quay lại tab Camera
    if (tabId === 'camera') {
        // Ép các camera card tính toán lại layout
        cameras.forEach(cam => {
            const canvas = document.getElementById(`canvas-${cam.id}`);
            if (canvas) {
                // Kích hoạt lại việc hiển thị nếu cần
                canvas.style.display = 'block';
            }
        });
    }

    // 3. Nếu chuyển sang tab lịch sử thì load dữ liệu
    if (tabId === 'history') {
        loadHistory();
    }
}
let cameras = [];
// Biến lưu trữ thời điểm cuối cùng lưu log cho mỗi camera để tránh lưu quá nhiều
let lastLogTime = {};

// =======================
// INIT APP
// =======================
window.onload = () => {
    initApp();
    loadHistory();
};

async function initApp() {
    try {
        const res = await fetch("/cameras");
        const data = await res.json();

        cameras = data.map(cam => ({
            id: cam.id,
            name: cam.name,
            url: cam.rtsp,
            ws: null,
            img: new Image(),
            deleted: false
        }));

        renderCameras();
    } catch (err) {
        console.error("❌ Load camera lỗi:", err);
    }
}

// =======================
// DRAW DETECTIONS
// =======================
function drawTracks(ctx, detections, canvasWidth, canvasHeight, originalImg) {
    if (!detections || !originalImg) return;

    const scaleX = canvasWidth / originalImg.width;
    const scaleY = canvasHeight / originalImg.height;

    detections.forEach(obj => {
        const [x1, y1, x2, y2] = obj.bbox;

        // scale đúng theo canvas
        const drawX = x1 * scaleX;
        const drawY = y1 * scaleY;
        const drawW = (x2 - x1) * scaleX;
        const drawH = (y2 - y1) * scaleY;

        // ===== VẼ BOX =====
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 2;
        ctx.strokeRect(drawX, drawY, drawW, drawH);

        // ===== LABEL =====
        //const label = `${obj.class} (${Math.round(obj.conf * 100)}%)`;
        const label = `${obj.class} ID:${obj.id}`;
        ctx.font = "bold 12px Arial";

        // đo kích thước text
        const textWidth = ctx.measureText(label).width;
        const textHeight = 14;

        // nền label
        ctx.fillStyle = "rgba(0, 255, 0, 0.8)";
        ctx.fillRect(drawX, drawY - textHeight, textWidth + 6, textHeight);

        // chữ
        ctx.fillStyle = "#000";
        ctx.fillText(
            label,
            drawX + 3,
            drawY - 3
        );
    });
}

// =======================
// FILTER HISTORY
// =======================
function filterHistory() {
    let input = document.getElementById('searchCameraName').value.toLowerCase();
    let historyItems = document.querySelectorAll('.history-item');

    historyItems.forEach(item => {
        let cameraNameEl = item.querySelector('.cam-name');
        if (cameraNameEl) {
            let nameText = cameraNameEl.textContent.toLowerCase();
            item.style.display = nameText.includes(input) ? "" : "none";
        }
    });
}
// =======================
// CAMERA RTSP SECTION
// =======================

// =======================
// ADD CAMERA
// =======================
async function addCamera() {
    const name = document.getElementById("camNameInput").value.trim();
    const url = document.getElementById("rtspInput").value.trim();

    if (!name || !url) {
        alert("Nhập đủ thông tin!");
        return;
    }

    try {
        const res = await fetch("/camera/add", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ name, rtsp: url })
        });

        const data = await res.json();

        const cam = {
            id: data.camera_id,
            name,
            url,
            ws: null,
            img: new Image(),
            deleted: false
        };

        cameras.push(cam);
        renderCameras();

    } catch (err) {
        console.error(err);
    }
}

// =======================
// REMOVE CAMERA
// =======================
async function removeCamera(id) {
    const cam = cameras.find(c => c.id === id);
    if (!cam) return;

    cam.deleted = true;
    if (cam.ws) cam.ws.close();

    document.getElementById(`cam-${id}`)?.remove();
    await fetch(`/camera/${id}`, { method: "DELETE" });

    cameras = cameras.filter(c => c.id !== id);
}

// =======================
// RENDER CAMERA
// =======================
function renderCameras() {
    const grid = document.getElementById("cameraGrid");
    grid.innerHTML = "";

    cameras.forEach(cam => {
        if (cam.deleted) return;

        const card = document.createElement("div");
        card.className = "camera-card";
        card.id = `cam-${cam.id}`;

        card.innerHTML = `
            <div class="camera-header">
                <span class="camera-name">Tên camrea: ${cam.name}</span>
                <button class="remove-camera">×</button>
            </div>
            <div class="canvas-wrapper">
                <canvas id="canvas-${cam.id}"></canvas>
            </div>
            
        `;

        card.querySelector(".remove-camera").onclick = () => removeCamera(cam.id);

        grid.appendChild(card);

        if (!cam.ws) {
            setTimeout(() => connectCameraWS(cam), 100);
        }
    });
}
// ==========================================
// CẤU HÌNH TỐC ĐỘ LƯU LOG (Đơn vị: Mili giây)
// ==========================================
const CONFIG = {
    SAVE_LOG_INTERVAL: 10000, // 5000ms = 5 giây. Muốn 10 giây thì sửa thành 10000.
};

// =======================
// CONNECT CAMERA WS
// =======================
function connectCameraWS(cam) {
    if (cam.deleted) return;

    // ❌ chặn multiple connection
    if (cam.ws &&
        (cam.ws.readyState === WebSocket.OPEN ||
         cam.ws.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const url = `${protocol}://${location.host}/ws/camera/${cam.id}`;

    cam.ws = new WebSocket(url);
    let reconnectTimer = null;
    let lastFrameTime = 0;

    cam.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.error) {
                removeCamera(cam.id);
                cam.ws.close();
                return;
            }

            const canvas = document.getElementById(`canvas-${cam.id}`);
            if (!canvas) return;

            const ctx = canvas.getContext("2d");

            // 🟢 throttle FPS render (anti lag)
            const now = Date.now();
            if (now - lastFrameTime < 33) return; // ~30 FPS
            lastFrameTime = now;

            const img = cam.img;

            // ❌ bỏ onload (fix lag + race condition)
            img.src = "data:image/jpeg;base64," + data.frame;

            img.onload = () => {
                const w = canvas.clientWidth || 320;
                const h = (img.height / img.width) * w;

                canvas.width = w;
                canvas.height = h;

                ctx.clearRect(0, 0, w, h);
                ctx.drawImage(img, 0, 0, w, h);

                const results = data.detections || [];

                if (typeof drawTracks === "function") {
                    drawTracks(ctx, results, w, h, img);
                }

                // count overlay
                ctx.fillStyle = "black";
                ctx.fillRect(5, 5, 80, 30);

                ctx.fillStyle = "#00FF00";
                ctx.font = "16px Arial";
                ctx.fillText(`👥 ${data.count}`, 10, 25);

                // save log (debounce)
                if ((data.count||0) > 0 && results.length > 0) {
                    const nowLog = Date.now();

                    if (
                        !lastLogTime[cam.id] ||
                        nowLog - lastLogTime[cam.id] > CONFIG.SAVE_LOG_INTERVAL
                    ) {
                        lastLogTime[cam.id] = nowLog;
                        saveDetectionLog(cam.id, data.frame, data.count, results);
                    }
                }
            };

        } catch (err) {
            console.error("WS message error:", err);
        }
    };

    // 🟢 FIX reconnect (NO SPAM)
    cam.ws.onclose = () => {
        cam.ws = null;

        if (cam.deleted) return;

        setTimeout(() => {
            if (!cam.ws || cam.ws.readyState === WebSocket.CLOSED) {
                connectCameraWS(cam);
            }
        }, 3000); // delay 3s (stable)
    };

    // 🟢 handle error
    cam.ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        cam.ws.close();
    };
}
// Hàm gửi log về server
// Thêm tham số detections vào định nghĩa hàm
// =======================
// SAVE LOG
// =======================
async function saveDetectionLog(camId, frame, count, detections) {
    try {
        await fetch("/camera/save_log", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                cam_id: camId,
                frame,
                count,
                detections
            })
        });
    } catch (err) {
        console.error(err);
    }
}
// Hàm load danh sách lịch sử từ server
async function loadHistory() {

    try {

        const res = await fetch("/camera/history");

        const logs = await res.json();

        const container = document.getElementById(
            "historyList"
        );

        if (!container) return;

        container.innerHTML = "";

        logs.forEach(log => {

            const card = document.createElement("div");

            card.className =
                "history-card history-item";

            const canvasId =
                `canvas-hist-${log.id}`;

            card.innerHTML = `

                <div class="canvas-wrapper">

                    <canvas
                        id="${canvasId}"
                        style="
                            width:100%;
                            border-radius:8px;
                        "
                    ></canvas>

                </div>

                <div class="info">
                    Phát hiện:
                    ${log.count} người
                </div>

                <div class="cam-name">
                    camera: ${log.camera_name}
                </div>

            `;

            container.appendChild(card);

            const canvas =
                document.getElementById(canvasId);

            if (!canvas) return;

            const ctx = canvas.getContext("2d");

            const img = new Image();

            // IMPORTANT
            img.src = log.image_url;

            img.onload = () => {

                const w =
                    canvas.clientWidth || 320;

                const h =
                    (img.height / img.width) * w;

                canvas.width = w;
                canvas.height = h;

                ctx.clearRect(0, 0, w, h);

                ctx.drawImage(
                    img,
                    0,
                    0,
                    w,
                    h
                );

                // IMPORTANT
                const results =
                    log.detections || [];

                drawTracks(
                    ctx,
                    results,
                    w,
                    h,
                    img
                );

                // overlay
                ctx.fillStyle = "black";

                ctx.fillRect(
                    5,
                    5,
                    90,
                    30
                );

                ctx.fillStyle = "#00FF00";

                ctx.font = "16px Arial";

                ctx.fillText(
                    `👥 ${log.count}`,
                    10,
                    25
                );
            };

            img.onerror = () => {

                console.error(
                    "Không load được ảnh:",
                    log.image_url
                );
            };
        });

    } catch (err) {

        console.error(
            "Lỗi history:",
            err
        );
    }
}

window.addEventListener("beforeunload", () => {
    cameras.forEach(cam => {
        if (cam.ws) {
            cam.ws.onclose = null;
            cam.ws.close();
            cam.ws = null;
        }
    });
});
// =======================
// IMAGE SECTION
// =======================
const imageInput = document.getElementById("imageInput");
const imageCanvas = document.getElementById("imageCanvas");
const ctxI = imageCanvas?.getContext("2d");

imageInput?.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/detect/image", { method: "POST", body: formData });
    const data = await res.json();
    const img = new Image();
    img.onload = () => {
        const maxWidth = 800;
        const scale = Math.min(1, maxWidth / img.width);
        imageCanvas.width = img.width * scale;
        imageCanvas.height = img.height * scale;
        ctxI.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);
        const results = data.detections || data.tracks || [];
        drawTracks(ctxI, results, imageCanvas.width, imageCanvas.height, img);
        document.getElementById("imageCount").innerHTML = `👥 Total Verified: <b>${data.count}</b>`;
    };
    img.src = data.image; 
});

// =======================
// VIDEO SECTION
// =======================
const videoInput = document.getElementById("videoInput");
const videoPlayer = document.getElementById("videoPlayer");
const canvasV = document.getElementById("videoCanvas");
const ctxV = canvasV?.getContext("2d");
const countV = document.getElementById("videoCount");

const hiddenCanvas = document.createElement("canvas");
const hiddenCtx = hiddenCanvas.getContext("2d");

let wsVideo;

function connectVideoWS() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    wsVideo = new WebSocket(`${protocol}://${location.host}/ws/video`);
    wsVideo.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const img = new Image();
        img.onload = () => {
            canvasV.width = videoPlayer.clientWidth;
            canvasV.height = videoPlayer.clientHeight;
            ctxV.clearRect(0, 0, canvasV.width, canvasV.height);
            ctxV.drawImage(img, 0, 0, canvasV.width, canvasV.height);
            const results = data.detections || data.tracks;
            drawTracks(ctxV, results, canvasV.width, canvasV.height, img);
            countV.innerHTML = `👥 Verified Count: <b style="color:#00FF00">${data.count}</b>`;
        };
        img.src = "data:image/jpeg;base64," + data.frame;
    };
    wsVideo.onclose = () => setTimeout(connectVideoWS, 2000);
}

videoInput?.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        videoPlayer.src = url;
        videoPlayer.play();
    }
});

videoPlayer?.addEventListener("play", () => {
    let last = 0;
    function sendFrame() {
        if (videoPlayer.paused || videoPlayer.ended) return;
        const now = Date.now();
        if (now - last < 66) { requestAnimationFrame(sendFrame); return; }
        last = now;
        if (videoPlayer.videoWidth > 0) {
            hiddenCanvas.width = videoPlayer.videoWidth;
            hiddenCanvas.height = videoPlayer.videoHeight;
            hiddenCtx.drawImage(videoPlayer, 0, 0);
            const dataURL = hiddenCanvas.toDataURL("image/jpeg", 0.6);
            if (wsVideo && wsVideo.readyState === WebSocket.OPEN) {
                wsVideo.send(JSON.stringify({ frame: dataURL }));
            }
        }
        requestAnimationFrame(sendFrame);
    }
    sendFrame();
});

connectVideoWS();