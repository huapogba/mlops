
-- 1. Bảng quản lý Camera
CREATE TABLE IF NOT EXISTS cameras (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    rtsp_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Bảng lưu Log sự kiện (mỗi lần chụp ảnh là 1 log)
CREATE TABLE IF NOT EXISTS camera_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    camera_id INT,
    person_count INT DEFAULT 0,
    head_count INT DEFAULT 0,
    image_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);

-- 3. Bảng lưu tọa độ chi tiết từng vật thể để train lại
CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    log_id INT,
    class_name VARCHAR(50),
    bbox_x FLOAT, -- Tọa độ center x
    bbox_y FLOAT, -- Tọa độ center y
    bbox_w FLOAT, -- Chiều rộng box
    bbox_h FLOAT, -- Chiều cao box
    confidence FLOAT,
    FOREIGN KEY (log_id) REFERENCES camera_logs(id) ON DELETE CASCADE
);

SELECT * FROM cameras;
SELECT * FROM camera_logs;
SELECT * FROM detections;
-- 1. Tắt kiểm tra khóa ngoại
SET FOREIGN_KEY_CHECKS = 0;

-- 2. Xóa dữ liệu trong từng bảng (Reset ID về 1)
 TRUNCATE TABLE detections;
 TRUNCATE TABLE camera_logs;
 TRUNCATE TABLE cameras;

-- 3. Bật lại kiểm tra khóa ngoại
 SET FOREIGN_KEY_CHECKS = 1;