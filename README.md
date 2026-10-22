# Hệ Thống Cảnh Báo Sớm Nguy Cơ Tiểu Đường (CO3117 - Machine Learning)

## Cấu trúc thư mục
```
project_diabetes/
├── data/
│   ├── raw/                 # Dữ liệu gốc tải từ Kaggle
│   └── processed/           # 2 bộ data (full biến & selected biến)
├── notebooks/
│   ├── 01_eda_feature_selection.ipynb   # Người 1
│   ├── 02_model_A.ipynb                 # Người 2
│   ├── 03_model_B.ipynb                 # Người 3
│   └── 04_evals_mlp_ensemble.ipynb      # Người 4
├── src/
│   ├── __init__.py
│   └── metrics.py           # Script metric chung (Người 4 viết sớm)
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
