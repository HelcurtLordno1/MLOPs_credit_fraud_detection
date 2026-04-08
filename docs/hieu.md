# Deployment & CI/CD Guide (Hieu)


## 🏗️ 1. Cấu trúc Triển khai (Deployment Architecture)
Phần triển khai được đóng gói bằng **Docker Compose** bao gồm 4 services chạy song song trên cùng 1 network ảo (`fraud-net`):
1. **app (FastAPI)**: Server phục vụ dự đoán thời gian thực. Bật trên cổng `8000`.
2. **mlflow**: Tracking server để quản lý model registry. Bật trên cổng `5000`.
3. **prometheus**: Thu thập log và metrics (Request count, latency histogram) từ `/metrics` của FastAPI. Bật trên cổng `9090`.
4. **grafana**: Dashboard để vẽ biểu đồ giám sát từ data của Prometheus. Bật trên cổng `3000`.

---

## 💻 2. Khởi chạy toàn hệ thống bằng Docker (Khuyên dùng)

**Yêu cầu:** Máy đã cài và đang bật sẵn Docker Desktop.

Đứng tại thư mục gốc của dự án, mở Terminal/CMD và chạy lệnh:
```bash
docker compose -f docker/docker-compose.yaml up --build -d
```
Quá trình build container sẽ mất khoảng 1-2 phút trong lần chạy đầu tiên.

### Truy cập các dịch vụ:
- **Test giao diện API (Swagger UI):** http://localhost:8000/docs
- **Quản lý Model (MLflow):** http://localhost:5000
- **Trang dữ liệu raw metrics (Prometheus):** http://localhost:9090
- **Bảng điều khiển (Grafana):** http://localhost:3000 *(Tài khoản mặc định: `admin` / `admin`)*

**Tắt hệ thống:**
```bash
docker compose -f docker/docker-compose.yaml down
```

---

## ⚡ 3. Chạy Server ở chế độ Local (Không dùng Docker)

Nếu không có Docker, bạn vẫn có thể bật Server API ảo trực tiếp trên máy bằng Uvicorn.

Bật môi trường (hoặc venv) và chạy:
```bash
uvicorn src.app.api:app --host 127.0.0.1 --port 8000 --reload
```
API sẽ Live tại `http://127.0.0.1:8000/docs`.

---

## 🤖 4. Tính năng Hot-Reload Model
Model `.joblib` được tải trực tiếp vào RAM. Trong suốt quá trình vận hành, nếu model mới được train lại hoặc param thay đổi (sinh ra file `latest.joblib` mới), **bạn không cần phải Restart API Server**.

Cách nạp Model mới mà không bị Disconnect:
- Trực tiếp chạy endpoint: `POST http://127.0.0.1:8000/reload` trên Swagger.
- API sẽ tự động refresh mô hình, các prediction tới sẽ được xử lý bằng Model mới nhất.

---

## 🧪 5. Automation Testing (CI)
Unit Test được viết bằng `pytest` dành riêng cho việc test API Endpoint (không cần model that) và Data Structure.

Chạy Local Test:
```bash
pytest tests/ -v
```

Hệ thống CI tự động được thiết lập thông qua Github Actions (`.github/workflows/ci.yaml`). Khi có bất kỳ lượt Push hay Pull Request nào lên nhánh `main`, Github sẽ tự động:
1. Chạy **Flake8** để check Linter.
2. Tiêm **Pytest** chạy kịch bản thử nghiệm API.
3. Chạy thử quy trình **Docker Build** xem Dockerfile có lỗi không.
