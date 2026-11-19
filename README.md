# Hệ Thống Cảnh Báo Sớm Nguy Cơ Tiểu Đường (CO3117 - Machine Learning)

Dự án Bài tập lớn môn **Học máy (CO3117)** — Trường Đại học Bách Khoa, ĐHQG-HCM.  
**Đề tài:** Xây dựng hệ thống cảnh báo sớm nguy cơ mắc bệnh tiểu đường dựa trên chỉ số sức khỏe và thói quen sinh hoạt (dữ liệu khảo sát **CDC BRFSS 2015**).

**Nhóm 23:**
* Nguyễn Bá Nam — MSSV: `2312167`
* Nguyễn Công Thành — MSSV: `2213133`
* Nguyễn Quang Hùng — MSSV: `2311297`
* Nguyễn Cảnh Nguyên — MSSV: `2312357`

---

## 1. Cấu trúc thư mục

```text
Machine_Learning/
├── data/
│   ├── raw/                             # 3 tệp dữ liệu gốc từ Kaggle (CDC BRFSS 2015)
│   ├── processed/                       # 2 bộ dữ liệu sạch (diabetes_full.csv & diabetes_selected.csv)
│   └── README.md                        # Mô tả chi tiết dữ liệu và từ điển 22 cột đặc trưng
├── notebooks/
│   ├── 01_eda_feature_selection.ipynb   # Tiền xử lý, EDA & Chọn lọc đặc trưng (Top 13)
│   ├── 02_model_A.ipynb                 # Mô hình A: Logistic Regression (L2, 5-fold CV)
│   ├── 03_model_B.ipynb                 # Mô hình B: Decision Tree (GridSearchCV, 5-fold CV)
│   ├── 04_MLP_model.ipynb               # Baseline Deep Learning: MLP PyTorch (Early Stopping)
│   ├── 05_ensemble_and_evaluation.ipynb # Soft Voting Ensemble, So sánh tổng hợp & Phân tích lỗi
│   └── figures/                         # Biểu đồ EDA xuất từ notebook 01 / src/plot_eda.py
├── src/
│   ├── __init__.py
│   ├── metrics.py                       # Nạp dữ liệu chuẩn (80:20, seed=42), tính metrics & tìm ngưỡng F2
│   └── plot_eda.py                      # Script độc lập vẽ & xuất 6 biểu đồ EDA
├── models/                              # Checkpoints (.joblib, .pt), xác suất (.npy/.csv) & bảng tổng hợp (.csv)
├── slides/                              # Slide thuyết trình nhóm
├── requirements.txt                     # Danh sách thư viện Python (pip)
├── Dockerfile                           # Cấu hình Docker để tái hiện môi trường độc lập
├── .dockerignore
└── README.md                            # Tài liệu hướng dẫn cài đặt, chạy code & ghi chú tái hiện
```

---

## 2. Yêu cầu và cài đặt môi trường

* **Hệ điều hành:** Linux / macOS / Windows
* **Python:** `>= 3.10` (khuyến nghị `Python 3.12`)
* **Lưu ý phiên bản thư viện:** `scikit-learn` được giới hạn `< 1.10.0` trong `requirements.txt` để tương thích với tham số `penalty` trong `02_model_A.ipynb`.

Có **2 cách** thiết lập môi trường (chọn 1 cách phù hợp):

### Cách 1: Python `venv` + `requirements.txt` (Khuyến nghị)

```bash
# 1. Tạo và kích hoạt môi trường ảo
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows (CMD / PowerShell)

# 2. Cài đặt các gói phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

### Cách 2: Docker (`Dockerfile`)

Image mặc định dùng **PyTorch CPU wheel** để gọn nhẹ và dễ chạy trên mọi máy:

```bash
# Build image (mặc định dùng PyTorch CPU)
docker build -t diabetes-ml .

# (Tùy chọn) Build với PyTorch CUDA từ PyPI nếu chạy trên máy có NVIDIA GPU:
# docker build --build-arg TORCH_INDEX_URL=https://pypi.org/simple -t diabetes-ml-gpu .

# Mở JupyterLab tại http://localhost:8888
docker run --rm -p 8888:8888 diabetes-ml
```

Để chạy toàn bộ pipeline không cần giao diện (headless) trong Docker và đồng bộ đầu ra về thư mục hiện tại:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  diabetes-ml bash -c "cd notebooks && \
    jupyter nbconvert --to notebook --execute --inplace \
      01_eda_feature_selection.ipynb \
      02_model_A.ipynb \
      03_model_B.ipynb \
      04_MLP_model.ipynb \
      05_ensemble_and_evaluation.ipynb"
```

---

## 3. Hướng dẫn chạy code

> **Quan trọng:** Các notebook sử dụng đường dẫn tương đối (`../data`, `../models`, `..`), vì vậy cần mở hoặc chạy notebook với **thư mục làm việc là `notebooks/`** (mặc định khi mở bằng JupyterLab/VS Code).

### 3.1. Tái hiện nhanh kết quả tổng hợp (~7 giây)

Khi trong `models/` đã có sẵn checkpoint (`logreg_*.joblib`, `decision_tree_*.joblib`) và mảng điểm dự đoán (`.npy`), chỉ cần chạy notebook cuối để tạo lại `models/all_models_comparison.csv`, đường cong ROC/PR, ma trận nhầm lẫn và phân tích lỗi:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 05_ensemble_and_evaluation.ipynb
```

Để xuất lại 6 biểu đồ EDA từ dữ liệu thô (`data/raw/`) vào `notebooks/figures/`:

```bash
# Chạy từ thư mục gốc Machine_Learning/
python src/plot_eda.py
```

### 3.2. Chạy lại toàn bộ quy trình từ đầu (~4 phút)

Mở `jupyter lab` và chạy lần lượt 5 notebook trong `notebooks/`, hoặc chạy tự động bằng dòng lệnh:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace \
  01_eda_feature_selection.ipynb \
  02_model_A.ipynb \
  03_model_B.ipynb \
  04_MLP_model.ipynb \
  05_ensemble_and_evaluation.ipynb
```

Chi tiết từng bước trong pipeline:

1. **Bước 1 — Tiền xử lý, EDA & Chọn lọc đặc trưng (`01_eda_feature_selection.ipynb`, ~23 giây):**
   * Đọc 3 tệp gốc trong `data/raw/`; dùng `diabetes_012_health_indicators_BRFSS2015.csv` làm nguồn chính.
   * Gộp nhóm *Tiền tiểu đường* (`1.0`) và *Tiểu đường* (`2.0`) thành nhãn `1`, lọc ngoại lai `BMI >= 70`, loại bỏ $4{,}096$ mẫu âm xung đột nhãn biên trên 21 biến.
   * Xuất 2 bộ dữ liệu sạch vào `data/processed/`: `diabetes_full.csv` ($248{,}985 \times 22$) và `diabetes_selected.csv` ($248{,}985 \times 14$, Top 13 đặc trưng).

2. **Bước 2 — Mô hình A: Logistic Regression (`02_model_A.ipynb`, ~49 giây):**
   * Đóng gói `StandardScaler` + `LogisticRegression` (`penalty='l2'`, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000`) trong `Pipeline`.
   * Tinh chỉnh siêu tham số $C$ và chọn ngưỡng $F_2$ trên xác suất out-of-fold bằng Stratified 5-Fold CV (`C = 0.001` cho cả Full và Selected).
   * Xuất `logreg_*.joblib` và `logreg_*_test_prob.npy` vào `models/`.

3. **Bước 3 — Mô hình B: Decision Tree (`03_model_B.ipynb`, ~106 giây):**
   * Khảo sát baseline không ràng buộc và Validation Curve theo `max_depth`.
   * GridSearchCV 5-fold Stratified CV trên lưới `max_depth`, `min_samples_leaf`, `criterion`, `ccp_alpha` (cố định `class_weight='balanced'`), chọn được `max_depth=8`, `min_samples_leaf=100`, `criterion='entropy'`, `ccp_alpha=0.0`.
   * Dò ngưỡng tối ưu theo $F_2$-score, xuất `decision_tree_*.joblib`, `y_prob_dt_*.npy`, `y_prob_dt_*.csv`, và `decision_tree_metrics.json` vào `models/`.

4. **Bước 4 — Baseline Deep Learning: MLP (`04_MLP_model.ipynb`, ~60 giây GPU / ~75 giây CPU):**
   * Kiến trúc PyTorch `Input(d) -> 64 -> 32 -> 1` với `BatchNorm1d`, `ReLU`, `Dropout(0.2)`.
   * Tách thêm 15% từ tập Train làm Validation (tỷ lệ chung 68% Train / 12% Val / 20% Test), chuẩn hóa bằng `StandardScaler` fit trên Train, dùng `BCEWithLogitsLoss(pos_weight ≈ 5.2427)` và `AdamW` với Early Stopping (`patience=4`).
   * Chọn ngưỡng $F_2$ trên tập Validation, đánh giá trên tập Test, xuất `mlp_*.pt` và `y_prob_mlp_*.npy` vào `models/`.

5. **Bước 5 — Ensemble Learning & Đánh giá tổng hợp (`05_ensemble_and_evaluation.ipynb`, ~7 giây):**
   * Kết hợp điểm dự đoán bằng **Equal Soft Voting** ($1/3$ mỗi mô hình) và **Weighted Soft Voting** ($0.35\,\text{LR} + 0.25\,\text{DT} + 0.40\,\text{MLP}$), chọn ngưỡng trên tập Validation của MLP.
   * Xuất `models/y_prob_ensemble_*.npy` và bảng so sánh tổng hợp `models/all_models_comparison.csv` (5 cấu hình $\times$ 2 bộ đặc trưng), vẽ ROC/PR, ma trận nhầm lẫn và phân tích lỗi.

---

## 4. Các ghi chú cần thiết về dữ liệu và tính tái hiện

* **Tính nhất quán của tập Test:** Tất cả các mô hình dùng chung `load_split_data` trong `src/metrics.py` (`test_size=0.20`, `stratify=y`, `random_state=42`) để so sánh trên đúng cùng $49{,}797$ mẫu kiểm thử ($41{,}820$ âm tính, $7{,}977$ dương tính).
* **Bộ chỉ số đánh giá & nguồn chọn ngưỡng:**
  * Do dữ liệu mất cân bằng ($16.02\%$ lớp dương) và ưu tiên giảm bỏ sót ca bệnh (False Negative), nhóm dùng **$F_2$-Score** (quét lưới $[0.05, 0.89]$, bước $0.01$) để chọn ngưỡng phân loại, đồng thời đánh giá xếp hạng bằng **Average Precision (AP, cột `PR_AUC` trong CSV)** và **ROC-AUC**.
  * Nguồn chọn ngưỡng: Logistic Regression dùng OOF 5-fold CV; MLP và Ensemble dùng tập Validation của MLP (LR và DT đã học trên các mẫu này nên kết quả Ensemble mang tính thăm dò); Decision Tree dùng tập Test (được ghi chú rõ trong báo cáo là kết quả thăm dò).
* **Lưu ý tái hiện giữa GPU và CPU (PyTorch MLP):**
  * Các bước `01`, `02`, `03` (NumPy/scikit-learn) tái hiện **chính xác 100%** trên cả GPU lẫn CPU.
  * Bước `04` (`04_MLP_model.ipynb`) được huấn luyện trên **NVIDIA CUDA GPU (GTX 1650)** với `seed=42`. Khi chạy lại trên cùng môi trường CUDA, toàn bộ điểm dự đoán và `models/all_models_comparison.csv` tái hiện **chính xác tuyệt đối (`max_abs_diff = 0.0`)**. Nếu chạy lại bước `04` trên **CPU-only** (hoặc Docker CPU), do khác biệt bộ sinh số ngẫu nhiên/dấu phẩy động giữa CPU và CUDA trong PyTorch, ngưỡng và các chỉ số của MLP/Ensemble có thể dao động rất nhỏ ($\approx \pm 0.001$–$0.002$ ở AP, ROC-AUC, $F_2$; xu hướng và kết luận giữ nguyên). Để đối chiếu đúng từng chữ số trong báo cáo trên máy không có GPU, chỉ cần chạy trực tiếp notebook `05` với các tệp `.npy` đã lưu sẵn trong `models/`.
* **Lưu ý đóng gói dung lượng lớn:** Các tệp trong `data/` (~70 MB) và `models/` (~10 MB) có dung lượng lớn. Nếu hệ thống nộp bài giới hạn kích thước file `.zip`, nhóm tách riêng thư mục `data/` (và/hoặc `models/`) lên Google Drive / GitHub public và đặt đường dẫn tải tại đây trước khi nộp.
