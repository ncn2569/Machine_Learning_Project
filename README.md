# Hệ Thống Cảnh Báo Sớm Nguy Cơ Tiểu Đường (CO3117 - Machine Learning)

Dự án Bài tập lớn môn Học máy (CO3117) - Trường Đại học Bách Khoa ĐHQG-HCM.  
Đề tài: **Xây dựng hệ thống cảnh báo sớm nguy cơ mắc bệnh tiểu đường dựa trên chỉ số sức khỏe và thói quen sinh hoạt (CDC BRFSS 2015)**.

---

## 1. Cấu trúc thư mục

```
Machine_Learning/
├── data/
│   ├── raw/                 # 3 tệp dữ liệu gốc từ Kaggle (CDC BRFSS 2015)
│   ├── processed/           # 2 bộ dữ liệu sạch (diabetes_full.csv & diabetes_selected.csv)
│   └── README.md            # Mô tả chi tiết dữ liệu và từ điển 22 cột đặc trưng
├── notebooks/
│   ├── 01_eda_feature_selection.ipynb   # Tiền xử lý, EDA & Chọn lọc đặc trưng (Người 1)
│   ├── 02_model_A.ipynb                 # Mô hình A: Logistic Regression (Người 2)
│   ├── 03_model_B.ipynb                 # Mô hình B: Decision Tree (Người 3)
│   ├── 04_MLP_model.ipynb               # Baseline Deep Learning: MLP PyTorch (Người 4)
│   └── 05_ensemble_and_evaluation.ipynb # Ensemble Learning, So sánh tổng hợp & SHAP (Người 4)
├── src/
│   ├── __init__.py
│   ├── metrics.py           # Module nạp dữ liệu chuẩn, đánh giá metrics & tìm ngưỡng F2
│   └── plot_eda.py          # Script vẽ & xuất biểu đồ phân tích dữ liệu
├── models/                  # Checkpoints mô hình (.joblib, .pt) và xác suất dự đoán (.npy)
├── slides/                  # Slide thuyết trình báo cáo
├── requirements.txt         # Danh sách thư viện phụ thuộc
└── README.md                # Hướng dẫn sử dụng & thông tin quan trọng
```

---

## 2. Cài đặt môi trường

Yêu cầu môi trường: **Python $\ge 3.10$** (khuyến nghị Python 3.10 – 3.12).

```bash
# 1. Tạo môi trường ảo
python -m venv .venv

# 2. Kích hoạt môi trường ảo
source .venv/bin/activate       # Trên Linux / macOS
# Hoặc trên Windows: .venv\Scripts\activate

# 3. Cài đặt các gói phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Hướng dẫn chạy thực nghiệm theo quy trình (Pipeline)

Quy trình thực nghiệm được tổ chức tuần tự qua 5 Jupyter Notebook trong thư mục `notebooks/`:

1. **Bước 1: Tiền xử lý & Chọn lọc đặc trưng (`01_eda_feature_selection.ipynb`)**
   * Làm sạch dữ liệu gốc từ `data/raw/diabetes_012_health_indicators_BRFSS2015.csv`.
   * Gộp nhóm Tiền tiểu đường và Tiểu đường thành nhãn nguy cơ `1`, lọc ngoại lai `BMI >= 70`, khử xung đột nhãn biên.
   * Xuất ra 2 bộ dữ liệu sạch trong `data/processed/`: `diabetes_full.csv` (21 biến) và `diabetes_selected.csv` (13 biến chọn lọc).

2. **Bước 2: Huấn luyện Mô hình A - Logistic Regression (`02_model_A.ipynb`)**
   * Huấn luyện mô hình phân loại tuyến tính với `StandardScaler`, `class_weight='balanced'`.
   * Tuning siêu tham số $C$ và penalty bằng Stratified 5-Fold Cross Validation.
   * Xuất mô hình và mảng xác suất dự đoán `logreg_*_test_prob.npy` vào `models/`.

3. **Bước 3: Huấn luyện Mô hình B - Decision Tree (`03_model_B.ipynb`)**
   * Huấn luyện cây quyết định có kiểm soát chiều sâu (`max_depth`, `min_samples_split`, pruning).
   * Dò tìm ngưỡng cắt xác suất tối ưu theo $F_2$-score.
   * Xuất mô hình và mảng xác suất dự đoán `y_prob_dt_*.npy` vào `models/`.

4. **Bước 4: Huấn luyện Baseline Deep Learning - MLP (`04_MLP_model.ipynb`)**
   * Xây dựng mạng nơ-ron truyền thẳng PyTorch (`Input -> 64 -> 32 -> 1`) với `BatchNorm1d`, `Dropout(0.2)`.
   * Sử dụng `BCEWithLogitsLoss` với `pos_weight = 5.24` giải quyết mất cân bằng lớp.
   * Dò tìm ngưỡng tối ưu trên tập Validation, đánh giá trên tập Test và xuất `y_prob_mlp_*.npy`, `mlp_*.pt` vào `models/`.

5. **Bước 5: Ensemble Learning & Đánh giá tổng hợp (`05_ensemble_and_evaluation.ipynb`)**
   * Nạp xác suất dự đoán từ `models/`, kết hợp theo cơ chế **Soft Voting** (Equal Voting và Weighted Voting: 0.35 LR + 0.25 DT + 0.40 MLP).
   * Tạo bảng so sánh tổng hợp 4 phương pháp trên cả 2 bộ dữ liệu (`models/all_models_comparison.csv`).
   * Trực quan hóa đường cong ROC, đường cong Precision-Recall, Confusion Matrix và phân tích đóng góp đặc trưng bằng **SHAP**.

---

## 4. Các thông tin quan trọng cần lưu ý

* **Bộ chỉ số đánh giá cốt lõi:**
  * Bài toán cảnh báo sớm y tế ưu tiên giảm thiểu ca bỏ sót bệnh (False Negative). Vì vậy, nhóm sử dụng **$F_2$-Score** (đặt trọng số Recall gấp đôi Precision) làm tiêu chí chính để chọn ngưỡng cắt xác suất (Optimal Threshold).
  * Do dữ liệu mất cân bằng (tỷ lệ bệnh ~16%), nhóm theo dõi chặt chẽ **PR-AUC (Precision-Recall AUC)** và **ROC-AUC** thay vì chỉ dựa vào Accuracy.
* **Tính nhất quán của dữ liệu Test:**
  * Tất cả các mô hình dùng chung hàm `load_split_data` từ `src/metrics.py` (Stratified Split 80:20, `random_state=42`) để đảm bảo tập Test hoàn toàn độc lập và giống nhau 100% giữa các thực nghiệm.
* **Tái hiện nhanh kết quả:**
  * Thư mục `models/` chứa sẵn xác suất dự đoán kiểm thử (`.npy`), cho phép chạy trực tiếp notebook `05_ensemble_and_evaluation.ipynb` để tái hiện bảng so sánh và đồ thị mà không cần phải huấn luyện lại từ đầu.
