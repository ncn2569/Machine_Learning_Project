# Hệ Thống Cảnh Báo Sớm Nguy Cơ Tiểu Đường (CO3117 - Machine Learning)

## Cấu trúc thư mục
```
project_diabetes/
├── data/
│   ├── README.md            # Tài liệu chi tiết 3 tệp raw, 2 tệp processed & từ điển 22 cột
│   ├── raw/                 # Dữ liệu gốc tải từ Kaggle (CDC BRFSS 2015)
│   └── processed/           # 2 bộ data sạch (full biến & selected biến)
├── notebooks/
│   ├── figures/                         # Bộ 9 biểu đồ EDA & tổng kết tiền xử lý
│   ├── 01_eda_feature_selection.ipynb   # Người 1
│   ├── 02_model_A.ipynb                 # Người 2
│   ├── 03_model_B.ipynb                 # Người 3
│   └── 04_evals_mlp_ensemble.ipynb      # Người 4
├── src/
│   ├── __init__.py
│   └── metrics.py           # Script metric chung
├── models/                  # Checkpoints mô hình (.pkl, .joblib, .pt)
├── slides/                  # File slide thuyết trình
├── requirements.txt
└── README.md
```

## Hướng dẫn cài đặt môi trường (Pip truyền thống)
```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Tổng quan các tệp dữ liệu (`data/`)

> Xem giải thích chi tiết từng tệp, từ điển 22 cột và hướng dẫn nạp dữ liệu tại **[`data/README.md`](./data/README.md)**.

### 1. Ba tệp dữ liệu thô gốc (`data/raw/` — CDC BRFSS 2015)
1. **`diabetes_012_health_indicators_BRFSS2015.csv`** (`253,680 × 22`, `21.68 MB`): **Tệp nguồn chính** gồm 3 lớp (`0.0`: Không bệnh - `84.24%`, `1.0`: Tiền tiểu đường - `1.83%`, `2.0`: Tiểu đường - `13.93%`).
2. **`diabetes_binary_health_indicators_BRFSS2015.csv`** (`253,680 × 22`, `21.68 MB`): Bản nhị phân trên Kaggle (21 biến đầu vào giống hệt bản `012`, nhưng tác giả gốc đã gộp `4,631` ca *Tiền tiểu đường* vào lớp `0.0` nên chỉ dùng để đối chiếu EDA).
3. **`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`** (`70,692 × 22`, `6.05 MB`): Bản lấy mẫu giảm nhân tạo 50:50 (`35,346` dòng đầu nhãn `0.0`, `35,346` dòng sau nhãn `1.0`; không dùng làm tập đánh giá thực tế).

### 2. Hai tệp dữ liệu sạch bàn giao (`data/processed/` — Phương án A, không chia sẵn Train/Test)
Được xuất từ `notebooks/01_eda_feature_selection.ipynb` (gộp `Tiền tiểu đường` + `Tiểu đường` thành `Diabetes_binary = 1`, lọc `599` dòng nhiễu `BMI >= 70`, khử `23,968` bản ghi trùng lặp và xử lý `1,804` dòng xung đột nhãn):

| Tệp dữ liệu | Kích thước (`Dòng × Cột`) | Dung lượng | Biến đặc trưng ($X$) | Phân phối nhãn `Diabetes_binary` |
| :--- | :---: | :---: | :--- | :--- |
| **`diabetes_full.csv`** | `227,309 × 22` | `10.12 MB` | **21 biến đầy đủ** | Lớp `0`: `187,745` (`82.59%`) / Lớp `1`: `39,564` (`17.41%`) |
| **`diabetes_selected.csv`** | `227,309 × 14` | `6.65 MB` | **Top 13 biến chọn lọc** (`GenHlth`, `HighBP`, `BMI`, `HighChol`, `Age`, `DiffWalk`, `PhysHlth`, `HeartDiseaseorAttack`, `Income`, `Education`, `PhysActivity`, `Stroke`, `MentHlth`) | Lớp `0`: `187,745` (`82.59%`) / Lớp `1`: `39,564` (`17.41%`) |
