# Dockerfile tái hiện thực nghiệm BTL Học máy (CO3117 - Nhóm 23)
FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Thư viện hệ thống tối thiểu cho matplotlib / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt phụ thuộc (mặc định dùng PyTorch CPU wheel để tối ưu dung lượng image)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url ${TORCH_INDEX_URL} "torch>=2.0.0" \
    && python -c "lines=[l.strip() for l in open('requirements.txt') if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('torch')]; open('/tmp/req_no_torch.txt','w').write('\n'.join(lines)+'\n')" \
    && pip install -r /tmp/req_no_torch.txt \
    && rm -f /tmp/req_no_torch.txt

# Sao chép mã nguồn, notebook, dữ liệu và mô hình
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY data/ ./data/
COPY models/ ./models/
COPY README.md ./

EXPOSE 8888

# Mặc định mở JupyterLab; có thể ghi đè để chạy headless bằng jupyter nbconvert
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token="]
