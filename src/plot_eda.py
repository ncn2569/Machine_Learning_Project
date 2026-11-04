"""
Script độc lập vẽ và xuất bộ 6 biểu đồ Khám phá Dữ liệu (EDA), Tiền xử lý & Chọn lọc Đặc trưng.
Đề tài: Hệ thống Cảnh báo sớm Nguy cơ mắc bệnh Tiểu đường (CO3117 - Học Máy)

Cách chạy từ thư mục gốc dự án:
    python src/plot_eda.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIG_DIR = PROJECT_ROOT / "notebooks" / "figures"

SELECTED_FEATURES = [
    "GenHlth",
    "HighBP",
    "BMI",
    "HighChol",
    "Age",
    "DiffWalk",
    "PhysHlth",
    "HeartDiseaseorAttack",
    "Income",
    "Education",
    "PhysActivity",
    "Stroke",
    "MentHlth",
]


def setup_plot_style() -> None:
    """Thiết lập style và font tiếng Việt thống nhất cho toàn bộ biểu đồ."""
    style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    plt.style.use(style)
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def load_and_clean_data(raw_dir: Path = RAW_DIR):
    """Đọc dữ liệu gốc BRFSS 2015, gộp nhãn cảnh báo sớm và thực hiện làm sạch."""
    df_012 = pd.read_csv(raw_dir / "diabetes_012_health_indicators_BRFSS2015.csv")
    orig_features = [c for c in df_012.columns if c != "Diabetes_012"]

    # 1. Gộp Tiền tiểu đường (1.0) và Tiểu đường (2.0) thành nhãn 1 (Nguy cơ / Mắc bệnh)
    df = df_012.copy()
    df["Diabetes_binary"] = (df["Diabetes_012"] >= 1.0).astype(np.int8)
    df = df.drop(columns=["Diabetes_012"])
    df = df[["Diabetes_binary"] + orig_features]
    for col in orig_features:
        df[col] = df[col].astype(np.int16 if col == "BMI" else np.int8)

    # 2. Lọc bỏ 599 bản ghi ngoại lai BMI >= 70
    df_bmi_clean = df[df["BMI"] < 70].copy()

    # 3. Giải quyết xung đột nhãn biên quyết định (ưu tiên giữ nhãn 1, giữ nguyên các mẫu cùng nhãn)
    max_label_per_profile = df_bmi_clean.groupby(orig_features, sort=False)["Diabetes_binary"].transform("max")
    conflict_neg_mask = (df_bmi_clean["Diabetes_binary"] == 0) & (max_label_per_profile == 1)
    df_clean = df_bmi_clean[~conflict_neg_mask].reset_index(drop=True)

    return df_012, df, df_bmi_clean, df_clean, orig_features


def compute_feature_ranking(df_clean: pd.DataFrame, orig_features: list[str]):
    """Tính VIF và xếp hạng 21 đặc trưng theo Pearson |r|, Mutual Information và Random Forest."""
    X_clean = df_clean[orig_features]
    y_clean = df_clean["Diabetes_binary"]

    X_std = (X_clean - X_clean.mean()) / X_clean.std()
    vif_vals = np.diag(inv(X_std.corr().values))

    pearson_r = X_clean.apply(lambda col: col.corr(y_clean))
    pearson_abs = pearson_r.abs()
    f_vals, _ = f_classif(X_clean, y_clean)

    samp = df_clean.sample(n=40000, random_state=42)
    mi_vals = mutual_info_classif(
        samp[orig_features], samp["Diabetes_binary"], discrete_features=True, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_clean, y_clean)
    rf_vals = rf.feature_importances_

    ranking_df = pd.DataFrame(
        {
            "VIF": vif_vals,
            "Pearson_r": pearson_r,
            "F_Stat": f_vals,
            "Pearson_Score": pearson_abs / pearson_abs.max() * 100,
            "MI_Score": mi_vals / mi_vals.max() * 100,
            "RF_Score": rf_vals / rf_vals.max() * 100,
        },
        index=orig_features,
    )
    ranking_df["Composite_Score"] = ranking_df[["Pearson_Score", "MI_Score", "RF_Score"]].mean(axis=1)
    ranking_df = ranking_df.sort_values("Composite_Score", ascending=False)

    dropped_features = [c for c in orig_features if c not in SELECTED_FEATURES]
    ranking_df["Decision"] = [
        "KEEP (Selected)" if f in SELECTED_FEATURES else "DROP (Low Signal)" for f in ranking_df.index
    ]
    return ranking_df, dropped_features


def plot_fig01_target_and_prediabetes(df_012: pd.DataFrame, out_dir: Path = FIG_DIR) -> Path:
    """Biểu đồ 1: Đối chiếu phân phối nhãn 3 tệp dữ liệu gốc & Hồ sơ lâm sàng nhóm Tiền tiểu đường."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), gridspec_kw={"width_ratios": [1.08, 1.25]})
    fig.suptitle(
        "Khảo sát Phân phối Nhãn Dữ liệu Thô & Cơ sở Lâm sàng Gộp nhóm Tiền Tiểu đường",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    datasets = [
        "1. diabetes_012\n(3 lớp gốc, N=253,680)",
        "2. diabetes_binary\n(Kaggle nhị phân, N=253,680)",
        "3. 5050split\n(Undersampled, N=70,692)",
        "4. Nhãn Cảnh báo sớm\n(Gộp 1.0 + 2.0, N=253,680)",
    ]
    p_healthy = [213703 / 253680 * 100, 218334 / 253680 * 100, 50.0, 213703 / 253680 * 100]
    p_prediab = [4631 / 253680 * 100, 0.0, 0.0, 0.0]
    p_diab = [35346 / 253680 * 100, 35346 / 253680 * 100, 50.0, 39977 / 253680 * 100]

    x_pos = np.arange(len(datasets))
    w = 0.54
    ax1.bar(x_pos, p_healthy, width=w, color="#2b7bba", edgecolor="black", linewidth=0.7, label="Lớp 0: Không bệnh")
    ax1.bar(
        x_pos,
        p_prediab,
        width=w,
        bottom=p_healthy,
        color="#f59e42",
        edgecolor="black",
        linewidth=0.7,
        label="Lớp 1 (gốc): Tiền tiểu đường",
    )
    ax1.bar(
        x_pos,
        p_diab,
        width=w,
        bottom=np.array(p_healthy) + np.array(p_prediab),
        color="#d62728",
        edgecolor="black",
        linewidth=0.7,
        label="Lớp Dương tính (Tiểu đường / Nguy cơ)",
    )

    for i in range(len(datasets)):
        ax1.text(
            i, p_healthy[i] / 2, f"{p_healthy[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5
        )
        if p_prediab[i] > 0:
            ax1.annotate(
                "1.83%\n(4,631)",
                xy=(i + w / 2, p_healthy[i] + p_prediab[i] / 2),
                xytext=(i + 0.48, 85.0),
                ha="center",
                va="center",
                fontsize=8.4,
                fontweight="bold",
                color="#b35900",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#fff8ee", edgecolor="#e68a00", lw=0.8),
                arrowprops=dict(arrowstyle="->", color="#b35900", lw=1.1),
            )
        top_center = p_healthy[i] + p_prediab[i] + p_diab[i] / 2
        ax1.text(
            i, top_center, f"{p_diab[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5
        )

    ax1.annotate(
        "Gộp nhầm 4,631 ca\nTiền TĐ vào Lớp 0!",
        xy=(1, 85.0),
        xytext=(1.0, 68.0),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color="#8a2be2",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8f0ff", edgecolor="#8a2be2", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#8a2be2", lw=1.1),
    )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets, fontsize=8.8)
    ax1.set_ylabel("Tỷ lệ phân phối (%)", fontsize=10.5)
    ax1.set_ylim(0, 108)
    ax1.set_title("A. So sánh cấu trúc nhãn giữa 3 tệp gốc & Nhãn mục tiêu", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

    df_temp = df_012.copy()
    df_temp["Obese_BMI30"] = (df_temp["BMI"] >= 30).astype(int)
    check_cols = ["HighBP", "HighChol", "Obese_BMI30", "DiffWalk", "HeartDiseaseorAttack"]
    check_labels = [
        "Huyết áp cao\n(HighBP)",
        "Mỡ máu cao\n(HighChol)",
        "Béo phì\n(BMI >= 30)",
        "Khó đi lại\n(DiffWalk)",
        "Bệnh tim mạch\n(HeartDisease)",
    ]

    grp_means = df_temp.groupby("Diabetes_012")[check_cols].mean() * 100
    x_ind = np.arange(len(check_cols))
    bw = 0.26

    rects0 = ax2.bar(
        x_ind - bw, grp_means.loc[0.0], width=bw, color="#2b7bba", edgecolor="black", linewidth=0.6, label="0: Không bệnh (N=213,703)"
    )
    rects1 = ax2.bar(
        x_ind, grp_means.loc[1.0], width=bw, color="#f59e42", edgecolor="black", linewidth=0.6, label="1: Tiền tiểu đường (N=4,631)"
    )
    rects2 = ax2.bar(
        x_ind + bw, grp_means.loc[2.0], width=bw, color="#d62728", edgecolor="black", linewidth=0.6, label="2: Tiểu đường (N=35,346)"
    )

    for rects, col_txt in [(rects0, "#1b4f72"), (rects1, "#b35900"), (rects2, "#990000")]:
        for r in rects:
            h = r.get_height()
            ax2.text(
                r.get_x() + r.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=col_txt
            )

    ax2.set_xticks(x_ind)
    ax2.set_xticklabels(check_labels, fontsize=9.2)
    ax2.set_ylabel("Tỷ lệ hiện diện đặc trưng (%)", fontsize=10.5)
    ax2.set_ylim(0, 86)
    ax2.set_title("B. Hồ sơ bệnh lý nền của Tiền tiểu đường rất gần nhóm Tiểu đường", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.8)

    out_path = out_dir / "fig01_target_distribution_and_prediabetes.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig02_continuous_and_bmi_outliers(df: pd.DataFrame, out_dir: Path = FIG_DIR) -> Path:
    """Biểu đồ 2: Phân tích nhóm biến Liên tục / Bán liên tục (BMI, PhysHlth, MentHlth) & Ngoại lai BMI >= 70."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))
    fig.suptitle(
        "Phân tích Nhóm Biến Liên tục / Bán liên tục (BMI, PhysHlth, MentHlth) & Phát hiện Ngoại lai",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    ax = axes[0, 0]
    bmi_0 = df.loc[(df["Diabetes_binary"] == 0) & (df["BMI"] <= 65), "BMI"]
    bmi_1 = df.loc[(df["Diabetes_binary"] == 1) & (df["BMI"] <= 65), "BMI"]
    bins_hist = np.arange(12, 66, 2)
    ax.hist(
        bmi_0,
        bins=bins_hist,
        density=True,
        alpha=0.55,
        color="#2b7bba",
        edgecolor="black",
        linewidth=0.5,
        label=f"Lớp 0: Khỏe mạnh (Mean={df.loc[df['Diabetes_binary']==0, 'BMI'].mean():.1f})",
    )
    ax.hist(
        bmi_1,
        bins=bins_hist,
        density=True,
        alpha=0.55,
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
        label=f"Lớp 1: Nguy cơ / Bệnh (Mean={df.loc[df['Diabetes_binary']==1, 'BMI'].mean():.1f})",
    )
    ax.axvline(25, color="#555555", linestyle="--", linewidth=1.3, label="Ngưỡng Thừa cân (BMI = 25)")
    ax.axvline(30, color="#111111", linestyle="-.", linewidth=1.5, label="Ngưỡng Béo phì (BMI = 30)")
    ax.set_xlabel("Chỉ số khối cơ thể - BMI (kg/m²)", fontsize=10)
    ax.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
    ax.set_title("A. Sự dịch chuyển phân phối BMI sang phải ở nhóm Nguy cơ / Bệnh", fontsize=11.2, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.8)

    ax = axes[0, 1]
    bmi_bins = [0, 18.5, 25, 30, 35, 40, 50, 60, 70, 200]
    bmi_labels = [
        "<18.5\n(Gầy)",
        "18.5-24.9\n(Bình thường)",
        "25-29.9\n(Thừa cân)",
        "30-34.9\n(Béo phì I)",
        "35-39.9\n(Béo phì II)",
        "40-49.9\n(Béo phì III)",
        "50-59.9\n(Rất nặng)",
        "60-69.9\n(Cực nặng)",
        ">=70\n(NGOẠI LAI)",
    ]
    bmi_cat = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_labels, right=False)
    bmi_stat = df.groupby(bmi_cat, observed=False)["Diabetes_binary"].agg(["count", "mean"])
    bmi_stat["pct"] = bmi_stat["mean"] * 100

    bar_colors = ["#5dade2", "#3498db", "#85c1e9", "#f8c471", "#f39c12", "#e67e22", "#cb4335", "#922b21", "#7b2cbf"]
    bars = ax.bar(np.arange(len(bmi_labels)), bmi_stat["pct"], color=bar_colors, edgecolor="black", linewidth=0.7, width=0.68)
    for idx, (b, cnt, pct) in enumerate(zip(bars, bmi_stat["count"], bmi_stat["pct"])):
        ax.text(
            b.get_x() + b.get_width() / 2,
            pct + 0.9,
            f"{pct:.1f}%\n(n={cnt:,})",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#7b2cbf" if idx == 8 else "#222222",
        )

    ax.annotate(
        "BMI >= 70 (n = 599): Tỷ lệ bệnh đột ngột rơi về 15.5%!\n(Tương đương nhóm Thừa cân -> Nhiễu / Lỗi nhập liệu)",
        xy=(8.0, 22.8),
        xytext=(2.85, 49.0),
        ha="center",
        fontsize=8.6,
        fontweight="bold",
        color="#5a189a",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f0fc", edgecolor="#7b2cbf", lw=1.1),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.18", color="#7b2cbf", lw=1.8),
    )
    ax.set_xticks(np.arange(len(bmi_labels)))
    ax.set_xticklabels(bmi_labels, fontsize=8.2)
    ax.set_ylabel("Tỷ lệ Nguy cơ / Mắc bệnh (%)", fontsize=10)
    ax.set_ylim(0, 58)
    ax.set_title("B. Tỷ lệ bệnh tăng đơn điệu theo BMI nhưng dị thường tại BMI >= 70", fontsize=11.2, fontweight="bold")

    day_bins = [-1, 0, 5, 14, 29, 30]
    day_names = ["0 ngày", "1 - 5 ngày", "6 - 14 ngày", "15 - 29 ngày", "30 ngày (Cả tháng)"]
    day_palette = ["#2b7bba", "#6baed6", "#fdae6b", "#f16913", "#cb181d"]

    for ax_sub, col_name, title_txt, ylim_top in [
        (axes[1, 0], "PhysHlth", "C. PhysHlth (Số ngày sức khỏe thể chất kém / 30 ngày)", 39),
        (axes[1, 1], "MentHlth", "D. MentHlth (Số ngày bất ổn tâm lý, stress / 30 ngày)", 33),
    ]:
        cat_s = pd.cut(df[col_name], bins=day_bins, labels=day_names)
        st = df.groupby(cat_s, observed=False)["Diabetes_binary"].agg(["count", "mean"])
        st["pct"] = st["mean"] * 100
        st["share"] = st["count"] / len(df) * 100
        x_d = np.arange(len(day_names))
        b_sub = ax_sub.bar(x_d, st["pct"], color=day_palette, edgecolor="black", linewidth=0.7, width=0.56)
        for b, cnt, pct, sh in zip(b_sub, st["count"], st["pct"], st["share"]):
            ax_sub.text(
                b.get_x() + b.get_width() / 2, pct + 0.7, f"{pct:.1f}%\n(n={cnt:,})", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )
        ax_sub.set_xticks(x_d)
        ax_sub.set_xticklabels([f"{nm}\n({sh:.1f}% mẫu)" for nm, sh in zip(day_names, st["share"])], fontsize=8.7)
        ax_sub.set_ylabel("Tỷ lệ Nguy cơ / Mắc bệnh (%)", fontsize=10)
        ax_sub.set_ylim(0, ylim_top)
        ax_sub.set_title(title_txt, fontsize=11.2, fontweight="bold")

    out_path = out_dir / "fig02_continuous_features_and_bmi_outliers.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig03_ordinal_risk_gradient(df_clean: pd.DataFrame, out_dir: Path = FIG_DIR) -> Path:
    """Biểu đồ 3: Sự phân hóa nguy cơ Tiểu đường theo 4 biến Thứ bậc trên tập dữ liệu sạch."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.8))
    fig.suptitle(
        f"Sự Phân hóa Nguy cơ Tiểu đường theo Nhóm Biến Thứ bậc (Trên Tập dữ liệu sạch N = {len(df_clean):,})",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    ax = axes[0, 0]
    gen_labels = ["1: Xuất sắc\n(Excellent)", "2: Rất tốt\n(Very Good)", "3: Tốt\n(Good)", "4: Khá/TB\n(Fair)", "5: Kém\n(Poor)"]
    gen_st = df_clean.groupby("GenHlth")["Diabetes_binary"].agg(["count", "mean"])
    gen_st["pct"] = gen_st["mean"] * 100
    gen_cols = ["#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    b_gen = ax.bar(np.arange(5), gen_st["pct"], color=gen_cols, edgecolor="black", linewidth=0.7, width=0.58)
    for b, cnt, pct in zip(b_gen, gen_st["count"], gen_st["pct"]):
        ax.text(b.get_x() + b.get_width() / 2, pct + 0.8, f"{pct:.1f}%\n(n={cnt:,})", ha="center", va="bottom", fontsize=8.6, fontweight="bold")
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(gen_labels, fontsize=9)
    ax.set_ylabel("Tỷ lệ Nguy cơ / Mắc bệnh (%)", fontsize=10)
    ax.set_ylim(0, 48)
    r_gen = df_clean["GenHlth"].corr(df_clean["Diabetes_binary"])
    ax.set_title(f"A. Tự đánh giá Sức khỏe Tổng quát (GenHlth) — Tương quan mạnh nhất (r = {r_gen:.3f})", fontsize=11.0, fontweight="bold")

    ax = axes[0, 1]
    age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
    age_st = df_clean.groupby("Age")["Diabetes_binary"].mean() * 100
    x_age = np.arange(1, 14)
    ax.plot(x_age, age_st.values, marker="o", color="#cb181d", linewidth=2.4, markersize=6.5)
    ax.fill_between(x_age, age_st.values, color="#fcbba1", alpha=0.45)
    for xv, yv in zip(x_age, age_st.values):
        ax.text(xv, yv + 0.8, f"{yv:.1f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#99000d")
    ax.set_xticks(x_age)
    ax.set_xticklabels(age_labels, rotation=28, fontsize=8.8)
    ax.set_ylabel("Tỷ lệ Nguy cơ / Mắc bệnh (%)", fontsize=10)
    ax.set_ylim(0, 29.5)
    r_age = df_clean["Age"].corr(df_clean["Diabetes_binary"])
    ax.set_title(f"B. Tỷ lệ bệnh theo 13 Nhóm Tuổi (Age, r = {r_age:.3f}) — Tăng gần 15 lần đến tuổi 70-74", fontsize=11.0, fontweight="bold")

    ax = axes[1, 0]
    inc_labels = ["1: <$10k", "2: $10-15k", "3: $15-20k", "4: $20-25k", "5: $25-35k", "6: $35-50k", "7: $50-75k", "8: >=$75k"]
    inc_st = df_clean.groupby("Income")["Diabetes_binary"].mean() * 100
    inc_cols = plt.cm.Blues(np.linspace(0.85, 0.35, 8))
    b_inc = ax.bar(np.arange(8), inc_st.values, color=inc_cols, edgecolor="black", linewidth=0.7, width=0.62)
    for b, pct in zip(b_inc, inc_st.values):
        ax.text(b.get_x() + b.get_width() / 2, pct + 0.6, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8.6, fontweight="bold")
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(inc_labels, rotation=20, fontsize=8.8)
    ax.set_ylabel("Tỷ lệ Nguy cơ / Mắc bệnh (%)", fontsize=10)
    ax.set_ylim(0, 34)
    r_inc = df_clean["Income"].corr(df_clean["Diabetes_binary"])
    ax.set_title(f"C. Thu nhập Hộ gia đình (Income) — Tương quan nghịch rõ rệt (r = {r_inc:.3f})", fontsize=11.0, fontweight="bold")

    ax = axes[1, 1]
    edu_labels = [
        "1: Chưa đi học",
        "2: Tiểu học/\nTHCS",
        "3: Chưa TN\nTHPT",
        "4: Tốt nghiệp\nTHPT/GED",
        "5: Cao đẳng/\nĐH dở dang",
        "6: Tốt nghiệp\nĐại học+",
    ]
    edu_st = df_clean.groupby("Education")["Diabetes_binary"].mean() * 100
    edu_cols = plt.cm.GnBu(np.linspace(0.85, 0.38, 6))
    b_edu = ax.bar(np.arange(6), edu_st.values, color=edu_cols, edgecolor="black", linewidth=0.7, width=0.58)
    for b, pct in zip(b_edu, edu_st.values):
        ax.text(b.get_x() + b.get_width() / 2, pct + 0.6, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(edu_labels, fontsize=8.6)
    ax.set_ylabel("Tỷ lệ Nguy cơ / Mắc bệnh (%)", fontsize=10)
    ax.set_ylim(0, 38)
    r_edu = df_clean["Education"].corr(df_clean["Diabetes_binary"])
    ax.set_title(f"D. Trình độ Học vấn (Education) — Tương quan nghịch (r = {r_edu:.3f})", fontsize=11.0, fontweight="bold")

    out_path = out_dir / "fig03_ordinal_features_risk_gradient.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig04_binary_risk_ratio(df_clean: pd.DataFrame, orig_features: list[str], out_dir: Path = FIG_DIR) -> Path:
    """Biểu đồ 4: So sánh Tỷ lệ mắc bệnh & Tỷ số Nguy cơ (Risk Ratio) của 14 Biến Nhị phân."""
    bin_cols = [c for c in orig_features if df_clean[c].nunique() == 2]
    bin_records = []
    for c in bin_cols:
        p0 = df_clean.loc[df_clean[c] == 0, "Diabetes_binary"].mean() * 100
        p1 = df_clean.loc[df_clean[c] == 1, "Diabetes_binary"].mean() * 100
        prev = df_clean[c].mean() * 100
        bin_records.append({"Feature": c, "Prevalence": prev, "P0": p0, "P1": p1, "RR": p1 / p0})

    bin_df = pd.DataFrame(bin_records).sort_values("RR", ascending=True).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.8, 7.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.suptitle(
        "Phân tích Tỷ lệ Nguy cơ / Mắc bệnh & Tỷ số Nguy cơ (Risk Ratio) trên 14 Biến Nhị phân",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    y_idx = np.arange(len(bin_df))
    bh = 0.37

    ax1.barh(y_idx - bh / 2, bin_df["P0"], height=bh, color="#7fb3d5", edgecolor="black", linewidth=0.6, label="Khi Đặc trưng = 0 (Không / Nữ)")
    ax1.barh(y_idx + bh / 2, bin_df["P1"], height=bh, color="#cb181d", edgecolor="black", linewidth=0.6, label="Khi Đặc trưng = 1 (Có / Nam)")
    for i, row in bin_df.iterrows():
        ax1.text(row["P0"] + 0.4, i - bh / 2, f"{row['P0']:.1f}%", va="center", fontsize=8.0, color="#1b4f72")
        ax1.text(row["P1"] + 0.4, i + bh / 2, f"{row['P1']:.1f}%", va="center", fontsize=8.2, fontweight="bold", color="#99000d")

    ax1.set_yticks(y_idx)
    ax1.set_yticklabels([f"{r['Feature']} ({r['Prevalence']:.1f}% mẫu=1)" for _, r in bin_df.iterrows()], fontsize=9.0)
    ax1.set_xlabel("Tỷ lệ Nguy cơ / Mắc bệnh (%)", fontsize=10)
    ax1.set_xlim(0, 41)
    ax1.set_title("A. Tỷ lệ bệnh khi Có (1) so với Không (0) đặc trưng", fontsize=11.3, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.8)

    rr_colors = ["#cb181d" if rr >= 2.0 else ("#f59e42" if rr >= 1.0 else "#2b7bba") for rr in bin_df["RR"]]
    ax2.axvline(1.0, color="#333333", linestyle="--", linewidth=1.2, zorder=1)
    ax2.barh(y_idx, bin_df["RR"], height=0.62, color=rr_colors, edgecolor="black", linewidth=0.6, zorder=2)
    for i, row in bin_df.iterrows():
        ax2.text(
            row["RR"] + 0.08,
            i,
            f"{row['RR']:.2f}x",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            zorder=3,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.85),
        )

    ax2.set_yticks(y_idx)
    ax2.set_yticklabels(bin_df["Feature"], fontsize=9.2)
    ax2.set_xlabel("Tỷ số Nguy cơ — Risk Ratio = P(Y=1 | X=1) / P(Y=1 | X=0)", fontsize=10)
    ax2.set_xlim(0, 5.9)
    ax2.set_title("B. Tỷ số Nguy cơ (Đỏ: Nguy cơ mạnh | Cam: Yếu | Xanh: Bảo vệ / Nhiễu ngược)", fontsize=11.3, fontweight="bold")

    out_path = out_dir / "fig04_binary_features_risk_ratio.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig05_correlation_and_feature_selection(
    df_clean: pd.DataFrame, ranking_df: pd.DataFrame, dropped_features: list[str], out_dir: Path = FIG_DIR
) -> Path:
    """Biểu đồ 5: Ma trận Tương quan Pearson & Xếp hạng Đặc trưng Đa tiêu chí."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.0, 8.2), gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.suptitle(
        "Ma trận Tương quan Pearson & Đánh giá Độ quan trọng Đặc trưng Đa tiêu chí (Feature Selection)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    ordered_cols = ["Diabetes_binary"] + list(ranking_df.index)
    corr_mat = df_clean[ordered_cols].corr().values
    mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    corr_masked = np.ma.array(corr_mat, mask=mask_upper)

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="white")
    im = ax1.imshow(corr_masked, cmap=cmap, vmin=-0.55, vmax=0.55, aspect="auto")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.038, pad=0.02)
    cbar.set_label("Hệ số tương quan Pearson (r)", fontsize=9.5)

    n_c = len(ordered_cols)
    for r_i in range(n_c):
        for c_i in range(r_i + 1):
            val = corr_mat[r_i, c_i]
            txt_col = "white" if abs(val) >= 0.32 else "#111111"
            ax1.text(c_i, r_i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=txt_col)

    ax1.set_xticks(np.arange(n_c))
    ax1.set_xticklabels(ordered_cols, rotation=45, ha="right", fontsize=8.0)
    ax1.set_yticks(np.arange(n_c))
    ax1.set_yticklabels(ordered_cols, fontsize=8.2)
    ax1.grid(False)
    ax1.set_title("A. Ma trận tương quan Pearson (Không có đa cộng tuyến, |r| cặp biến <= 0.52)", fontsize=11.0, fontweight="bold")

    rk_plot = ranking_df.iloc[::-1]
    y_rk = np.arange(len(rk_plot))
    bh = 0.25

    ax2.barh(y_rk - bh, rk_plot["Pearson_Score"], height=bh, color="#2b7bba", label="Tương quan tuyến tính |r|")
    ax2.barh(y_rk, rk_plot["MI_Score"], height=bh, color="#f59e42", label="Mutual Information (Phi tuyến)")
    ax2.barh(y_rk + bh, rk_plot["RF_Score"], height=bh, color="#d62728", label="Random Forest Gini Importance")

    cutoff_y = len(dropped_features) - 0.5
    ax2.axhline(cutoff_y, color="#111111", linestyle="--", linewidth=1.6)
    ax2.text(32, cutoff_y + 0.35, "▲ Top 13 Biến quan trọng nhất (Đưa vào diabetes_selected.csv)", color="#1f7a1f", fontsize=8.8, fontweight="bold")
    ax2.text(32, cutoff_y - 0.75, "▼ Nhóm 8 Biến tín hiệu yếu / nhiễu (Lược bỏ ở bộ Selected)", color="#b30000", fontsize=8.8, fontweight="bold")

    ax2.set_yticks(y_rk)
    ax2.set_yticklabels(rk_plot.index, fontsize=9.0, fontweight="bold")
    ax2.set_xlabel("Điểm quy chuẩn tương đối (Thang 0 - 100, Max = 100)", fontsize=10)
    ax2.set_xlim(0, 108)
    ax2.set_title("B. Xếp hạng 21 Đặc trưng theo 3 thước đo độc lập", fontsize=11.2, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.8)

    out_path = out_dir / "fig05_correlation_and_feature_selection.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig06_preprocessing_pipeline_summary(
    df: pd.DataFrame,
    df_bmi_clean: pd.DataFrame,
    df_clean: pd.DataFrame,
    ranking_df: pd.DataFrame,
    dropped_features: list[str],
    out_dir: Path = FIG_DIR,
) -> Path:
    """Biểu đồ 6: Tổng kết Quy trình Làm sạch Dữ liệu & Điểm Tổng hợp Chọn lọc Đặc trưng."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.05, 1.15]})
    fig.suptitle(
        "Tổng kết Quy trình Tiền xử lý & Bàn giao 2 Bộ Dữ liệu Sạch (diabetes_full.csv & diabetes_selected.csv)",
        fontsize=13.8,
        fontweight="bold",
        y=1.02,
    )

    stages = [
        f"1. Dữ liệu gốc gộp nhãn\n(N = {len(df):,})",
        f"2. Lọc ngoại lai BMI < 70\n(N = {len(df_bmi_clean):,})",
        f"3. Xử lý xung đột nhãn\nBàn giao (N = {len(df_clean):,})",
    ]
    c0_counts = [
        int((df["Diabetes_binary"] == 0).sum()),
        int((df_bmi_clean["Diabetes_binary"] == 0).sum()),
        int((df_clean["Diabetes_binary"] == 0).sum()),
    ]
    c1_counts = [
        int((df["Diabetes_binary"] == 1).sum()),
        int((df_bmi_clean["Diabetes_binary"] == 1).sum()),
        int((df_clean["Diabetes_binary"] == 1).sum()),
    ]
    totals = [a + b for a, b in zip(c0_counts, c1_counts)]

    x_st = np.arange(len(stages))
    bw_st = 0.34
    ax1.bar(
        x_st - bw_st / 2, c0_counts, width=bw_st, color="#2b7bba", edgecolor="black", linewidth=0.7, label="Lớp 0: Khỏe mạnh (No Diabetes)"
    )
    ax1.bar(
        x_st + bw_st / 2,
        c1_counts,
        width=bw_st,
        color="#d62728",
        edgecolor="black",
        linewidth=0.7,
        label="Lớp 1: Nguy cơ / Mắc bệnh (Pre + Diabetes)",
    )

    for i in range(len(stages)):
        ax1.text(
            x_st[i] - bw_st / 2,
            c0_counts[i] + 3500,
            f"{c0_counts[i]:,}\n({c0_counts[i]/totals[i]*100:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#1b4f72",
        )
        ax1.text(
            x_st[i] + bw_st / 2,
            c1_counts[i] + 3500,
            f"{c1_counts[i]:,}\n({c1_counts[i]/totals[i]*100:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#99000d",
        )

    ax1.set_xticks(x_st)
    ax1.set_xticklabels(stages, fontsize=9.2)
    ax1.set_ylabel("Số lượng mẫu (Bản ghi)", fontsize=10.5)
    ax1.set_ylim(0, 268000)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.set_title("A. Biến động số lượng mẫu qua các bước làm sạch", fontsize=11.3, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.8)

    rk_rev = ranking_df.iloc[::-1]
    y_pos = np.arange(len(rk_rev))
    bar_cols = ["#2ca02c" if d.startswith("KEEP") else "#d9534f" for d in rk_rev["Decision"]]
    bars_comp = ax2.barh(y_pos, rk_rev["Composite_Score"], height=0.66, color=bar_cols, edgecolor="black", linewidth=0.6)
    for b, val in zip(bars_comp, rk_rev["Composite_Score"]):
        ax2.text(val + 1.0, b.get_y() + b.get_height() / 2, f"{val:.1f}", va="center", fontsize=8.4, fontweight="bold")

    cutoff_idx = len(dropped_features) - 0.5
    ax2.axhline(cutoff_idx, color="black", linestyle="--", linewidth=1.4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(rk_rev.index, fontsize=9.0)
    ax2.set_xlabel("Điểm tổng hợp chuẩn hóa — Composite Score (Trung bình |r|, MI, RF)", fontsize=10)
    ax2.set_xlim(0, 111)
    ax2.set_title("B. Top 13 biến Selected (Xanh lá) vs 8 biến Lược bỏ (Đỏ)", fontsize=11.3, fontweight="bold")

    out_path = out_dir / "fig06_preprocessing_pipeline_summary.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_eda_figures(raw_dir: Path = RAW_DIR, out_dir: Path = FIG_DIR) -> list[Path]:
    """Chạy toàn bộ pipeline nạp dữ liệu, xếp hạng đặc trưng và xuất 6 đồ thị EDA."""
    setup_plot_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. Đang đọc và làm sạch dữ liệu từ:", raw_dir)
    df_012, df, df_bmi_clean, df_clean, orig_features = load_and_clean_data(raw_dir)
    print(f"   -> Dữ liệu sạch: {df_clean.shape[0]:,} dòng x {df_clean.shape[1]} cột")

    print("2. Đang tính VIF và xếp hạng đa tiêu chí 21 đặc trưng...")
    ranking_df, dropped_features = compute_feature_ranking(df_clean, orig_features)

    print("3. Đang vẽ và xuất 6 biểu đồ vào:", out_dir)
    saved_paths = [
        plot_fig01_target_and_prediabetes(df_012, out_dir),
        plot_fig02_continuous_and_bmi_outliers(df, out_dir),
        plot_fig03_ordinal_risk_gradient(df_clean, out_dir),
        plot_fig04_binary_risk_ratio(df_clean, orig_features, out_dir),
        plot_fig05_correlation_and_feature_selection(df_clean, ranking_df, dropped_features, out_dir),
        plot_fig06_preprocessing_pipeline_summary(df, df_bmi_clean, df_clean, ranking_df, dropped_features, out_dir),
    ]
    for idx, p in enumerate(saved_paths, 1):
        print(f"   [{idx}/6] Đã lưu: {p.name}")
    return saved_paths


if __name__ == "__main__":
    generate_all_eda_figures()

