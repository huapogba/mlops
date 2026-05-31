# SCRIPT THUYẾT TRÌNH ĐỒ ÁN: HỆ THỐNG PHÁT HIỆN VÀ THEO DÕI ĐÁM ĐÔNG

## SLIDE 1: TIÊU ĐỀ & GIỚI THIỆU CHUNG

**Tiêu đề:** Hệ Thống Phát Hiện và Theo Dõi Đám Đông Thời Gian Thực

Xin chào! Hôm nay em xin được thuyết trình về một hệ thống thông minh về phát hiện và theo dõi đám đông sử dụng trí tuệ nhân tạo.

Đồ án này là một ứng dụng thực tế có thể được sử dụng trong:
- **Quản lý sự kiện** - Giám sát quy mô đám đông tại các sự kiện ngoài trời
- **An ninh công cộng** - Cảnh báo khi có tập trung đông người ở những khu vực nhạy cảm
- **Giao thông** - Theo dõi tình hình giao thông, người đi bộ tại các giao lộ
- **Quản lý nhà ga, sân bay** - Kiểm soát lưu lượng hành khách
- **Giám sát công ty** - Theo dõi tập trung người ở những khu vực làm việc

---

## SLIDE 2: TỔNG QUAN KIẾN TRÚC HỆ THỐNG

**Tiêu đề:** Kiến Trúc Hệ Thống

Hệ thống được thiết kế theo kiến trúc ba tầng:

**1. Tầng Frontend (Giao diện người dùng):**
- Là một ứng dụng web được phát triển bằng HTML, CSS, JavaScript
- Cho phép người dùng xem trực tiếp video stream từ camera
- Hiển thị số lượng người được phát hiện, ID của từng cá nhân được theo dõi
- Cung cấp các thống kê và biểu đồ phân tích

**2. Tầng Backend (Xử lý logic):**
- Được phát triển bằng FastAPI (framework Python hiệu suất cao)
- Xử lý các yêu cầu từ phía frontend
- Chạy model AI để phát hiện người, theo dõi chuyển động
- Kết nối với database để lưu dữ liệu
- Tích hợp với Prometheus để giám sát hiệu năng hệ thống

**3. Tầng Database:**
- Sử dụng MySQL để lưu trữ toàn bộ dữ liệu
- Lưu lịch sử phát hiện, thông tin người dùng, log hệ thống
- Hỗ trợ truy vấn thời gian thực

**4. Hệ thống Bên Ngoài:**
- **MLflow** - Theo dõi các thí nghiệm huấn luyện model, quản lý phiên bản model
- **DVC** - Quản lý phiên bản dữ liệu huấn luyện
- **Prometheus + Grafana** - Giám sát hiệu năng hệ thống
- **Docker** - Containerize toàn bộ hệ thống để dễ triển khai

---

## SLIDE 3: CÁC CHỨC NĂNG CHÍNH

**Tiêu đề:** Chức Năng Chính của Hệ Thống

Hệ thống có 4 chức năng chính:

**1. Phát Hiện Người (Object Detection)**
- Nhận diện con người từ video stream hoặc ảnh
- Vẽ hộp bao quanh từng người được phát hiện (bounding box)
- Cung cấp độ tin cậy (confidence score) cho mỗi lần phát hiện
- Có thể phát hiện cả toàn cơ thể và đầu của người

**2. Theo Dõi Đám Đông (Crowd Tracking)**
- Gán ID duy nhất cho mỗi cá nhân được phát hiện
- Theo dõi chuyển động của mỗi người theo thời gian
- Đếm tổng số người hiện tại trong khung hình
- Tính toán tổng số người duy nhất đã xuất hiện

**3. Huấn Luyện Model Tự Động (Auto-Training)**
- Khi có dữ liệu mới được thu thập, hệ thống tự động huấn luyện lại model
- Hệ thống quyết định có nên huấn luyện model mới hay không dựa vào các metrics
- Lưu trữ phiên bản model cũ để có thể quay lại nếu cần
- Giảm thiểu công việc thủ công của kỹ sư

**4. Giám Sát & Cảnh Báo (Monitoring & Alerts)**
- Theo dõi hiệu năng hệ thống (FPS, độ trễ xử lý, v.v.)
- Cảnh báo khi số lượng người vượt quá ngưỡng cho phép
- Ghi nhận lại các sự kiện bất thường
- Cung cấp các báo cáo thống kê theo thời gian

---

## SLIDE 4: KIẾN THỨC KỸ THUẬT ÁP DỤNG

**Tiêu đề:** Các Kỹ Thuật AI/ML Được Sử Dụng

**1. YOLOv8 (You Only Look Once)**
- Là mô hình deep learning tiên tiến cho phát hiện đối tượng
- Ưu điểm: 
  - Tốc độ xử lý cực nhanh (real-time)
  - Độ chính xác cao
  - Dễ triển khai
  - Có nhiều kích cỡ model từ nano đến xlarge
- Chúng tôi sử dụng YOLOv8n (bản nano) để tối ưu hóa tốc độ

**2. ByteTrack / BotsORT (Multi-Object Tracking)**
- Giải pháp theo dõi nhiều đối tượng từ từng frame khác nhau
- Sử dụng các đặc trưng:
  - IoU (Intersection over Union) - đo độ trùng lặp giữa các bounding box
  - Appearance feature - đặc điểm hình ảnh của từng người
  - Kalman Filter - dự đoán vị trí tiếp theo của đối tượng
- Cho phép gán ID liên tục cho cùng một người qua nhiều frame

**3. Data Version Control (DVC)**
- Quản lý phiên bản dữ liệu huấn luyện (tương tự Git nhưng cho dữ liệu)
- Theo dõi những thay đổi trong tập dữ liệu
- Hỗ trợ tái tạo lại các thí nghiệm cũ với cùng dữ liệu

**4. MLflow - Experiment Tracking**
- Ghi nhận chi tiết từng thí nghiệm huấn luyện
- Lưu trữ hyperparameters, metrics, model artifacts
- So sánh hiệu năng giữa các phiên bản model khác nhau
- Quản lý vòng đời model (staging → production)

**5. WebSocket - Real-time Communication**
- Cho phép gửi video stream và kết quả phát hiện trực tiếp từ server tới client
- Cập nhật giao diện người dùng theo thời gian thực mà không cần refresh trang

---

## SLIDE 5: QUY TRÌNH XỬ LÝ DỮ LIỆU

**Tiêu đề:** Quy Trình Xử Lý - Từ Dữ Liệu Đầu Vào Đến Kết Quả Đầu Ra

**Bước 1: Thu Thập & Chuẩn Bị Dữ Liệu**
- Video stream từ camera được gửi tới server
- Chia nhỏ video thành các frame riêng lẻ
- Resize ảnh về kích thước chuẩn (640x640) để phù hợp với model

**Bước 2: Phát Hiện Người**
- Model YOLOv8 xử lý từng frame
- Trả về danh sách bounding box + confidence score
- Lọc bỏ các phát hiện có độ tin cậy thấp (< ngưỡng)

**Bước 3: Theo Dõi & Gán ID**
- ByteTrack algorithm nhận input từ bước 2
- So sánh vị trí, hình dáng người với frame trước
- Gán ID (ID_1, ID_2, ID_3...) cho từng cá nhân
- Cập nhật track history

**Bước 4: Tính Toán Metrics**
- Đếm số người hiện tại
- Cộng dồn số người duy nhất đã từng xuất hiện
- Tính FPS (frame per second)
- Ghi nhận thời gian xử lý

**Bước 5: Lưu Trữ & Hiển Thị**
- Lưu kết quả vào database MySQL
- Gửi dữ liệu tới frontend qua WebSocket
- Frontend hiển thị video annotated (có vẽ bounding box) cho người dùng

**Bước 6: Quyết Định Huấn Luyện**
- Hệ thống đánh giá chất lượng model hiện tại
- Nếu metrics giảm (drift detection), tự động huấn luyện model mới
- Lưu trữ model tốt nhất, loại bỏ model kém

---

## SLIDE 6: CHI TIẾT QUY TRÌNH HUẤN LUYỆN

**Tiêu đề:** Quy Trình Huấn Luyện Tự Động (Auto-Training Pipeline)

**Giai Đoạn 1: Chuẩn Bị Dữ Liệu Huấn Luyện**
- Tập hợp các ảnh & annotation từ dữ liệu mới
- Chia thành 3 phần: train (70%), validation (20%), test (10%)
- Tạo file dataset.yaml với các đường dẫn và label
- Áp dụng augmentation (rotation, flip, brightness adjustment) để tăng độ đa dạng

**Giai Đoạn 2: Huấn Luyện Model**
- Load pre-trained model YOLOv8n từ model trước đó
- Sử dụng hyperparameters tối ưu:
  - Epochs: 1-10 (tuỳ thuộc vào lượng dữ liệu)
  - Image size: 640x640
  - Batch size: 16
  - Learning rate: 0.01
- MLflow ghi nhận toàn bộ chi tiết quá trình
- Mỗi epoch có một checkpoint model

**Giai Đoạn 3: Đánh Giá Hiệu Năng**
- Chạy model trên test set
- Tính toán các metrics:
  - **Precision** - Tỷ lệ phát hiện đúng
  - **Recall** - Tỷ lệ tìm thấy được người
  - **mAP** (mean Average Precision) - Chỉ số chính để đánh giá
  - **Inference time** - Thời gian xử lý một ảnh

**Giai Đoạn 4: Quyết Định Model**
- So sánh metrics của model mới với model cũ
- Nếu model mới tốt hơn: chấp nhận và deploy
- Nếu model mới kém hơn: giữ nguyên model cũ
- Lưu trữ 3 phiên bản model gần nhất trong /model
- MLflow giúp dễ dàng quay lại phiên bản cũ nếu cần

**Giai Đoạn 5: Drift Detection**
- Theo dõi xem model có bị "drift" (chất lượng giảm) không
- Nếu phát hiện drift, hệ thống tự động trigger huấn luyện lại
- Ghi nhận baseline performance để so sánh

---

## SLIDE 7: GIAO DIỆN NGƯỜI DÙNG & TRỰC QUAN HÓA

**Tiêu đề:** Giao Diện & Các Tính Năng Tương Tác

**Trang Chính (Dashboard):**
- Hiển thị video stream thời gian thực từ camera
- Vẽ bounding box và ID cho mỗi người được phát hiện
- Hiện thị số lượng người hiện tại ở góc trên cùng
- Cập nhật mức độ tín hiệu (signal strength)

**Thanh Thông Tin Thời Gian Thực:**
- **Current People Count** - Số người hiện tại trong khung hình
- **Total Unique People** - Tổng số người duy nhất đã từng xuất hiện
- **FPS** - Tốc độ xử lý (frame per second), thường từ 15-30 FPS
- **Inference Latency** - Thời gian xử lý một frame (ms)
- **Model Version** - Phiên bản model hiện tại đang chạy

**Lịch Sử & Thống Kê:**
- Biểu đồ số lượng người theo giờ/ngày
- Bảng thống kê chi tiết các sự kiện phát hiện
- Khả năng export báo cáo thành PDF

**Cài Đặt & Quản Lý:**
- Điều chỉnh ngưỡng tin cậy (confidence threshold)
- Chọn model hiện tại (switch giữa các phiên bản)
- Xem lịch sử huấn luyện
- Restart hoặc dừng hệ thống

---

## SLIDE 8: KIẾN TRÚC TRIỂN KHAI (DEPLOYMENT)

**Tiêu đề:** Cách Hệ Thống Được Triển Khai & Vận Hành

**Containerization với Docker:**
- Toàn bộ hệ thống được đóng gói thành Docker container
- Mỗi thành phần chạy trong một container riêng:
  - **Backend** - FastAPI server
  - **Frontend** - Web server
  - **Database** - MySQL database
  - **MLflow** - Model tracking server
  - **Prometheus** - Metrics collection

**Docker Compose Orchestration:**
- File docker-compose.yaml định nghĩa toàn bộ dịch vụ
- Một lệnh duy nhất khởi động cả hệ thống: `docker-compose up`
- Tự động quản lý network giữa các container
- Dễ dàng scale up (thêm nhiều camera) hoặc scale down

**Giám Sát Hệ Thống (Monitoring Stack):**
- **Prometheus** - Thu thập metrics từ hệ thống
  - CPU, memory usage
  - Inference requests total
  - FPS, inference latency
  - Model version hiện tại
- **Grafana** - Tạo dashboard trực quan hóa dữ liệu từ Prometheus
  - Hiển thị trend theo thời gian
  - Alert khi có anomaly
  - Real-time dashboard

**Workflow Tự Động (DVC Pipeline):**
- 4 bước chính được tự động hóa:
  1. Train - Huấn luyện model
  2. Evaluate - Đánh giá model
  3. Decide - Quyết định deploy hay không
  4. Deploy - Cập nhật model vào production
- Mỗi bước có dependencies rõ ràng
- Có thể trigger thủ công hoặc tự động dựa trên điều kiện

---

## SLIDE 9: ỨNG DỤNG THỰC TẾ & LỢI ÍCH

**Tiêu đề:** Các Ứng Dụng Thực Tế & Lợi Ích Của Hệ Thống

**1. Quản Lý Sự Kiện & An Ninh**
- Cảnh báo khi số lượng người vượt quá ngưỡng an toàn
- Tạo bản đồ nhiệt (heat map) để hiểu hành vi đám đông
- Phát hiện tập trung bất thường ở những khu vực không mong muốn

**2. Tối Ưu Hóa Giao thông**
- Phân tích luồng người đi bộ tại các giao lộ
- Điều chỉnh đèn giao thông dựa trên tải người
- Dự báo tắc đường bằng cách theo dõi mật độ

**3. Phân Tích Khách Hàng (Retail Analytics)**
- Đếm lượng khách vào cửa hàng
- Phân tích hành vi khách hàng (dừng lại ở đâu, bao lâu)
- Tối ưu hóa layout cửa hàng dựa trên dữ liệu

**4. Quản Lý Công Trình**
- Kiểm soát số lượng công nhân tại các khu vực nhạy cảm
- Phát hiện khu vực tập trung đông mà không đủ an toàn

**Lợi Ích Chính:**
✓ **Tự động hóa** - Giảm công nhân giám sát thủ công
✓ **Thời gian thực** - Phát hiện và cảnh báo ngay lập tức
✓ **Khả năng học hỏi** - Model tự cải thiện qua thời gian
✓ **Tiết kiệm chi phí** - Một camera thay thế nhiều nhân viên
✓ **Dữ liệu hữu ích** - Cung cấp insights sâu sắc cho quyết định kinh doanh

---

## SLIDE 10: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN TƯƠNG LAI

**Tiêu đề:** Kết Luận & Các Hướng Phát Triển

**Thành Tựu Chính:**
✓ Xây dựng hệ thống phát hiện & theo dõi đám đông hoạt động tốt
✓ Triển khai kiến trúc microservices với Docker
✓ Tích hợp auto-training pipeline
✓ Giám sát hệ thống toàn bộ với Prometheus + Grafana
✓ Giao diện người dùng thân thiện

**Các Thách Thức Đã Giải Quyết:**
- Đảm bảo inference tốc độ cao (real-time processing)
- Quản lý phiên bản model và dữ liệu hiệu quả
- Tự động hóa quyết định deploy model
- Giám sát và alert khi model performance giảm

**Hướng Phát Triển Tương Lai:**
1. **Mở Rộng Multi-Camera**
   - Tích hợp nhiều camera cùng lúc
   - Theo dõi cùng một người qua nhiều camera
   - Cross-camera trajectory analysis

2. **Phát Hiện Nâng Cao**
   - Phát hiện hành vi bất thường (cấu thế, chạy, ngã)
   - Phân loại độ tuổi, giới tính
   - Phát hiện vật phẩm được mang (ba lô, balo, v.v.)

3. **Tối Ưu Hóa Hiệu Năng**
   - Sử dụng model nhẹ hơn (YOLO-NAS, Mobile-SAM)
   - Triển khai trên edge devices (TPU, Jetson)
   - Giảm latency để xử lý video 4K

4. **Tích Hợp AI/ML Tiên Tiến**
   - Sử dụng Large Language Models cho phân tích hành vi
   - Anomaly detection với unsupervised learning
   - Reinforcement learning cho tối ưu hóa resource

5. **Cải Thiện UX**
   - Dashboard mobile-responsive
   - Real-time alerts via SMS/Email/App
   - Mobile app để kiểm soát từ xa

**Kết Luận:**
Hệ thống này chứng minh rằng kỹ thuật AI hiện đại, đặc biệt là YOLO, có thể được ứng dụng thực tế để giải quyết các bài toán thực từ quản lý sự kiện, an ninh, cho tới phân tích kinh doanh. Sự kết hợp giữa machine learning models, MLOps practices, và cloud-native architecture tạo ra một hệ thống đáng tin cậy, có thể mở rộng, và tự cải thiện.

---

## GỢI Ý TRÌNH BÀY:

**Thời gian mỗi slide:** 1.5 - 2 phút
- Slide 1: 1 phút (giới thiệu)
- Slide 2-3: 2 phút mỗi slide (tổng quan & chức năng)
- Slide 4-5: 2 phút mỗi slide (kỹ thuật & quy trình)
- Slide 6-8: 2 phút mỗi slide (chi tiết)
- Slide 9-10: 1.5 phút mỗi slide (ứng dụng & kết luận)

**Tổng cộng:** ~15 phút thuyết trình (nếu là 10 slide)

**Demo được khuyến khích:**
- Chạy hệ thống live và hiển thị kết quả phát hiện trực tiếp
- Mở MLflow để hiển thị lịch sử huấn luyện model
- Hiển thị Prometheus/Grafana dashboard
- Mở Docker dashboard để giải thích kiến trúc containerized

