# Tài Liệu Mô Tả Chi Tiết Các Tệp Dữ Liệu (`data/`)

Thư mục này chứa toàn bộ dữ liệu gốc (`data/raw/`) và 2 bộ dữ liệu sạch đã chọn lọc đặc trưng (`data/processed/`) phục vụ bài toán **Hệ thống Cảnh báo sớm Nguy cơ mắc bệnh Tiểu đường dựa trên Chỉ số Sức khỏe và Thói quen Sinh hoạt** (Học phần CO3117 - Học Máy).

> **Lưu ý Git:** Hai bộ dữ liệu sạch bàn giao trong `data/processed/` (`diabetes_full.csv` và `diabetes_selected.csv`) **đã được đưa sẵn lên Git** để các thành viên pull về huấn luyện mô hình ngay. Riêng 3 tệp dữ liệu thô dung lượng lớn trong `data/raw/` vẫn được bỏ qua trong `.gitignore` (chỉ cần tải từ Kaggle đặt vào `data/raw/` nếu muốn chạy lại từ đầu `notebooks/01_eda_feature_selection.ipynb`).

---

## 1. Chi Tiết Từng Tệp Trong 3 Tệp Dữ Liệu Thô (`data/raw/`)

Cả 3 tệp dữ liệu thô đều được tác giả Alex Teboul trích xuất và làm sạch sơ bộ từ cuộc khảo sát hành vi sức khỏe qua điện thoại **CDC BRFSS 2015** (*Behavioral Risk Factor Surveillance System* của Hoa Kỳ, khảo sát gốc có 441,456 người và hơn 330 biến; sau khi loại các dòng thiếu câu trả lời trên 22 biến cốt lõi thì còn **253,680 bản ghi**).

Đặc điểm chung của cả 3 tệp raw:
- **Không có giá trị khuyết (`NaN` / `Null` = 0)** trên toàn bộ các cột.
- Toàn bộ **22 cột** đều lưu dưới kiểu số thực `float64` (`0.0`, `1.0`, `28.0`...), trong đó **cột đầu tiên** là biến mục tiêu (nhãn bệnh), **21 cột tiếp theo** giữ nguyên tên và thứ tự đặc trưng đầu vào.

---

### 1.1. Tệp 1: `diabetes_012_health_indicators_BRFSS2015.csv` *(Tệp nguồn chính của dự án)*

- **Kích thước & Dung lượng:** `253,680 dòng × 22 cột` (`22,738,151 bytes` $\approx$ `21.68 MB`).
- **Biến mục tiêu:** `Diabetes_012` (Phân loại **3 lớp** phản ánh đầy đủ tiến trình bệnh lý từ câu hỏi khảo sát `DIABETE3` của CDC):
  - **`0.0` — Không mắc tiểu đường** *(hoặc chỉ từng bị tiểu đường thai kỳ)*: **`213,703` mẫu (`84.24%`)**.
  - **`1.0` — Tiền tiểu đường (`Prediabetes`)** *(hoặc đường huyết cận biên)*: **`4,631` mẫu (`1.83%`)**.
  - **`2.0` — Mắc bệnh Tiểu đường (`Diabetes`)**: **`35,346` mẫu (`13.93%`)**.
- **Đặc điểm chất lượng dữ liệu bên trong:**
  - Có **`23,899` dòng trùng lặp hoàn toàn** (`9.42%` tổng dữ liệu) trên cả 22 cột.
  - Nếu chỉ xét 21 cột đặc trưng đầu vào, có `227,908` tổ hợp đặc trưng duy nhất (`25,772` dòng bị trùng tổ hợp đặc trưng), trong đó có **`1,834` tổ hợp xung đột nhãn** (cùng 21 chỉ số giống hệt nhau nhưng xuất hiện ở nhiều lớp bệnh khác nhau).
  - Biến `BMI` trải từ `12.0` đến `98.0` (trung bình `28.38`, trung vị `27.0`), chứa **`599` bản ghi dị thường `BMI >= 70`** có tỷ lệ tiểu đường đột ngột rơi từ `42.8%` về `13.5%` do lỗi quy đổi đơn vị/nhập liệu.
- **Vai trò trong dự án:** **ĐƯỢC CHỌN LÀM NGUỒN CHÍNH** vì là tệp duy nhất tách bạch rõ `4,631` ca **Tiền tiểu đường (`1.0`)**, cho phép nhóm gộp `1.0` + `2.0` thành nhóm **Có nguy cơ / Mắc bệnh (`1`)** đúng với yêu cầu **Hệ thống Cảnh báo sớm**.

---

### 1.2. Tệp 2: `diabetes_binary_health_indicators_BRFSS2015.csv` *(Bản nhị phân mất cân bằng trên Kaggle)*

- **Kích thước & Dung lượng:** `253,680 dòng × 22 cột` (`22,738,154 bytes` $\approx$ `21.68 MB` — lớn hơn tệp `012` đúng 3 bytes do tên cột tiêu đề `"Diabetes_binary"` dài hơn `"Diabetes_012"` 3 ký tự).
- **Quan hệ với Tệp 1 (`diabetes_012`):** Toàn bộ **21 cột đặc trưng của `253,680` dòng giống hệt 100% theo đúng từng dòng** với tệp `diabetes_012`. Các chỉ số thống kê đặc trưng (`BMI` trung bình `28.38`, `HighBP = 42.90%`, `HighChol = 42.41%`, `GenHlth = 2.51`, `Age = 8.03`, `599` mẫu `BMI >= 70`) hoàn toàn trùng khớp.
- **Biến mục tiêu:** `Diabetes_binary` (Phân loại **2 lớp**, mất cân bằng tỷ lệ `~6.18 : 1`):
  - **`0.0` — Không tiểu đường:** **`218,334` mẫu (`86.07%`)**.
  - **`1.0` — Tiểu đường:** **`35,346` mẫu (`13.93%`)**.
- **Phát hiện lỗi ánh xạ nhãn quan trọng:**
  - Mặc dù tài liệu mô tả trên Kaggle của tác giả ghi *"0 is for no diabetes, and 1 is for prediabetes or diabetes"*, kiểm tra chéo (`pd.crosstab`) cho thấy thực tế tệp này đã **gộp toàn bộ `4,631` bệnh nhân Tiền tiểu đường (`Diabetes_012 == 1.0`) vào lớp `0.0`** (`213,703 + 4,631 = 218,334`), và chỉ giữ `Diabetes_012 == 2.0` làm lớp `1.0` (`35,346`).
  - Do gộp `1.0` vào `0.0`, số bản ghi trùng lặp hoàn toàn tăng nhẹ lên **`24,206` dòng (`9.54%`)** và có **`1,566` tổ hợp đặc trưng xung đột nhãn** (liên quan `5,218` dòng).
- **Vai trò trong dự án:** **Chỉ dùng để đối chiếu trong bước EDA**, không dùng làm nguồn huấn luyện vì việc gán nhãn `0` (Khỏe mạnh) cho `4,631` người Tiền tiểu đường (vốn có `HighBP = 62.9%`, `HighChol = 62.1%`, `BMI = 30.72`) làm nhiễu biên quyết định của hệ thống cảnh báo sớm.

---

### 1.3. Tệp 3: `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` *(Bản lấy mẫu cân bằng nhân tạo 50:50)*

- **Kích thước & Dung lượng:** `70,692 dòng × 22 cột` (`6,347,570 bytes` $\approx$ `6.05 MB`).
- **Biến mục tiêu:** `Diabetes_binary` (Phân loại **2 lớp** cân bằng tuyệt đối `1 : 1`):
  - **`0.0` — Không tiểu đường:** **`35,346` mẫu (`50.00%`)**.
  - **`1.0` — Tiểu đường:** **`35,346` mẫu (`50.00%`)**.
- **Cách tệp này được tạo ra & 4 nhược điểm kỹ thuật cần lưu ý:**
  1. **Cấu trúc lấy mẫu giảm (Random Undersampling):** Tác giả giữ nguyên 100% `35,346` ca Tiểu đường (`Diabetes_012 == 2.0`) và lấy ngẫu nhiên `35,346` mẫu từ `218,334` mẫu lớp `0.0` của tệp `diabetes_binary` (bỏ đi `182,988` mẫu khỏe mạnh, tức đánh mất `83.8%` thông tin lớp đa số).
  2. **Lẫn bệnh nhân Tiền tiểu đường trong lớp `0.0`:** Vì lấy mẫu từ lớp `0.0` của `diabetes_binary`, lớp `0.0` của tệp `5050split` bị lẫn ít nhất **`680` hồ sơ đặc trưng riêng của bệnh nhân Tiền tiểu đường** mang nhãn "Không bệnh".
  3. **Dữ liệu xếp thứ tự đơn điệu theo nhãn (Chưa xáo trộn):** `35,346` dòng đầu tiên (index `0` $\rightarrow$ `35,345`) hoàn toàn là nhãn `0.0`; `35,346` dòng cuối (index `35,346` $\rightarrow$ `70,691`) hoàn toàn là nhãn `1.0`. Vẫn còn `1,635` dòng trùng lặp (`2.31%`), `408` tổ hợp xung đột nhãn và `154` dòng ngoại lai `BMI >= 70`.
  4. **Lệch phân phối quần thể (Prior & Covariate Shift):** Do tỷ lệ người bệnh bị đẩy từ `13.93%` lên `50.00%`, trung bình toàn bộ các biến bệnh nền trong tệp này cao bất thường so với dân số thật (`HighBP` vọt từ `42.90%` lên `56.35%`, `HighChol` từ `42.41%` lên `52.57%`, `BMI` từ `28.38` lên `29.86`, `GenHlth` từ `2.51` lên `2.84`).
- **Vai trò trong dự án:** **Không sử dụng để huấn luyện/đánh giá chính thức**, vì đánh giá trên tập Test 50:50 sẽ làm phóng đại giả tạo Precision và PR-AUC, đồng thời gây báo động giả (False Positive) hàng loạt khi triển khai thực tế.

---

## 2. Quy Trình Làm Sạch Tạo Ra 2 Bộ Dữ Liệu Bàn Giao (`data/processed/`)

Từ tệp nguồn **`diabetes_012_health_indicators_BRFSS2015.csv`** (`253,680` dòng), notebook `notebooks/01_eda_feature_selection.ipynb` thực hiện 4 bước làm sạch nghiêm ngặt (không chia sẵn Train/Test):

1. **Gộp nhãn theo Phương án A (Chuẩn Cảnh báo sớm):**
   - `Diabetes_binary = 0`: Không mắc bệnh (`Diabetes_012 == 0.0` $\rightarrow$ `213,703` mẫu, `84.24%`).
   - `Diabetes_binary = 1`: Tiền tiểu đường + Tiểu đường (`Diabetes_012 in {1.0, 2.0}` $\rightarrow$ `4,631 + 35,346 = 39,977` mẫu, `15.76%`).
2. **Lọc bỏ ngoại lai lỗi nhập liệu `BMI >= 70`:** Loại bỏ `599` mẫu nhiễu (`BMI` từ `70` đến `98`), giữ miền giá trị y sinh hợp lệ `BMI` $\in [12, 69]$, còn `253,081` mẫu.
3. **Khử trùng lặp hoàn toàn (`drop_duplicates`):** Loại bỏ `23,968` bản ghi trùng lặp hoàn toàn để chống rò rỉ dữ liệu khi chia Train/Test về sau, còn `229,113` mẫu.
4. **Giải quyết xung đột nhãn biên quyết định:** Loại bỏ `1,804` bản ghi âm tính có cùng 21 giá trị đặc trưng trùng khít 100% với bản ghi dương tính (ưu tiên giữ nhãn `1` để làm sạch ranh giới quyết định và bảo toàn độ nhạy cảnh báo sớm), đồng thời ép kiểu toàn bộ cột về số nguyên (`int8` / `int16`).

---

## 3. Chi Tiết 2 Tệp Dữ Liệu Sạch Bàn Giao Trong `data/processed/`

Thư mục `data/processed/` chỉ gồm đúng **2 tệp chuẩn** theo yêu cầu phân công (không chia sẵn Train/Test; hai tệp chứa **cùng `227,309` bệnh nhân theo đúng thứ tự dòng 1-1**):

| Tên tệp | Kích thước (`Dòng × Cột`) | Dung lượng | Biến mục tiêu (Cột 0) | Số biến đặc trưng ($X$) | Phân phối nhãn sạch | Kiểm định chất lượng |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **`diabetes_full.csv`** | `227,309 × 22` | `10.12 MB` | `Diabetes_binary` | **21 biến đầy đủ** | • Lớp `0`: **187,745** (`82.59%`)<br>• Lớp `1`: **39,564** (`17.41%`) | `0` NaN, `0` Inf, `0` dòng trùng lặp, `0` xung đột nhãn trên 21 biến, `BMI` $\in [12, 69]$. |
| **`diabetes_selected.csv`** | `227,309 × 14` | `6.65 MB` | `Diabetes_binary` | **13 biến chọn lọc** | • Lớp `0`: **187,745** (`82.59%`)<br>• Lớp `1`: **39,564** (`17.41%`) | Khớp 100% từng dòng và nhãn với `diabetes_full.csv` (giảm 38% số chiều, giữ $>91\%$ tổng độ quan trọng đặc trưng). |

---

## 4. Từ Điển Dữ Liệu 22 Cột & Đối Chiếu Bộ `Full` vs `Selected`

| STT | Tên cột | Kiểu dữ liệu | Miền giá trị | Có trong `diabetes_full.csv` | Có trong `diabetes_selected.csv` | Giải thích ý nghĩa & Lý do Giữ / Lược bỏ |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- |
| 0 | **`Diabetes_binary`** | `int8` | `{0, 1}` | ✅ *(Target)* | ✅ *(Target)* | **Nhãn mục tiêu:** `0` = Khỏe mạnh; `1` = Tiền tiểu đường hoặc Tiểu đường. |
| 1 | **`GenHlth`** | `int8` | `{1..5}` | ✅ | ✅ *(Top 1)* | Tự đánh giá sức khỏe tổng quát: `1` (Xuất sắc) $\rightarrow$ `5` (Kém). Biến dự báo mạnh nhất. |
| 2 | **`HighBP`** | `int8` | `{0, 1}` | ✅ | ✅ *(Top 2)* | Chẩn đoán huyết áp cao (`0` = Không, `1` = Có). |
| 3 | **`BMI`** | `int16` | `[12, 69]` | ✅ | ✅ *(Top 3)* | Chỉ số khối cơ thể ($\text{kg/m}^2$), đã làm sạch ngoại lai $\ge 70$. |
| 4 | **`HighChol`** | `int8` | `{0, 1}` | ✅ | ✅ *(Top 4)* | Chẩn đoán mỡ máu / cholesterol cao (`0` = Không, `1` = Có). |
| 5 | **`Age`** | `int8` | `{1..13}` | ✅ | ✅ *(Top 5)* | 13 nhóm tuổi (mỗi bậc 5 năm): `1` (18–24 tuổi), `2` (25–29), ..., `13` ($\ge 80$ tuổi). |
| 6 | **`DiffWalk`** | `int8` | `{0, 1}` | ✅ | ✅ *(Top 6)* | Gặp khó khăn nghiêm trọng khi đi bộ hoặc leo cầu thang (`0` = Không, `1` = Có). |
| 7 | **`PhysHlth`** | `int8` | `[0, 30]` | ✅ | ✅ *(Top 7)* | Số ngày sức khỏe thể chất không tốt (ốm/chấn thương) trong 30 ngày qua. |
| 8 | **`HeartDiseaseorAttack`** | `int8` | `{0, 1}` | ✅ | ✅ *(Top 8)* | Tiền sử bệnh mạch vành (CHD) hoặc nhồi máu cơ tim (MI) (`0` = Không, `1` = Có). |
| 9 | **`Income`** | `int8` | `{1..8}` | ✅ | ✅ *(Top 9)* | Thu nhập hộ gia đình: `1` ($<\$10\text{k}$/năm) $\rightarrow$ `8` ($\ge \$75\text{k}$/năm). |
| 10 | **`Education`** | `int8` | `{1..6}` | ✅ | ✅ *(Top 10)* | Trình độ học vấn: `1` (Chưa đi học) $\rightarrow$ `6` (Tốt nghiệp Đại học trở lên). |
| 11 | **`PhysActivity`** | `int8` | `{0, 1}` | ✅ | ✅ *(Top 11)* | Có tập thể dục / vận động thể chất ngoài công việc trong 30 ngày qua (`0` = Không, `1` = Có). |
| 12 | **`Stroke`** | `int8` | `{0, 1}` | ✅ | ✅ *(Top 12)* | Tiền sử bị đột quỵ (`0` = Không, `1` = Có). |
| 13 | **`MentHlth`** | `int8` | `[0, 30]` | ✅ | ✅ *(Top 13)* | Số ngày sức khỏe tâm thần không tốt (stress, trầm cảm) trong 30 ngày qua. |
| 14 | **`CholCheck`** | `int8` | `{0, 1}` | ✅ | ❌ *(Lược bỏ)* | Kiểm tra cholesterol trong 5 năm qua (Biến hành chính gần như hằng số: $>96\%$ bằng 1). |
| 15 | **`HvyAlcoholConsump`** | `int8` | `{0, 1}` | ✅ | ❌ *(Lược bỏ)* | Uống rượu bia nặng (Nhiễu ngược *Sick Stopper Effect* — người bệnh phải kiêng rượu). |
| 16 | **`Smoker`** | `int8` | `{0, 1}` | ✅ | ❌ *(Lược bỏ)* | Đã hút $\ge 100$ điếu thuốc trong đời (Tín hiệu phân tách yếu trên MI & RF Importance). |
| 17 | **`Veggies`** | `int8` | `{0, 1}` | ✅ | ❌ *(Lược bỏ)* | Ăn rau xanh $\ge 1$ lần/ngày (Đóng góp $< 0.8\%$ Gini Importance). |
| 18 | **`Sex`** | `int8` | `{0, 1}` | ✅ | ❌ *(Lược bỏ)* | Giới tính: `0` = Nữ, `1` = Nam (Tương quan rất thấp $|r| = 0.031$). |
| 19 | **`Fruits`** | `int8` | `{0, 1}` | ✅ | ❌ *(Lược bỏ)* | Ăn trái cây $\ge 1$ lần/ngày ($|r| = 0.025$, tín hiệu dự báo rất yếu). |
| 20 | **`NoDocbcCost`** | `int8` | `{0, 1}` | ✅ | ❌ *(Lược bỏ)* | Cần gặp bác sĩ trong 12 tháng qua nhưng không đi được vì chi phí ($|r| = 0.026$). |
| 21 | **`AnyHealthcare`** | `int8` | `{0, 1}` | ✅ | ❌ *(Lược bỏ)* | Có bảo hiểm y tế (Biến gần như hằng số: $>95\%$ bằng 1, độ quan trọng thấp nhất). |

---

## 5. Quy Ước Đọc Dữ Liệu Thống Nhất Cho Người 2, Người 3 và Người 4

Vì `diabetes_full.csv` và `diabetes_selected.csv` không chia sẵn Train/Test nhưng **có cùng thứ tự `227,309` dòng**, các thành viên huấn luyện mô hình chỉ cần dùng chung tham số `test_size=0.20, stratify=y, random_state=42` là sẽ thu được **cùng một tập Test cố định** (giúp Người 4 ghép `y_prob` làm Voting/Stacking Ensemble khớp 100%):

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc bộ Full (hoặc đổi thành "../data/processed/diabetes_selected.csv" cho bộ Selected)
df = pd.read_csv("../data/processed/diabetes_full.csv")
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Tách Train/Test thống nhất toàn nhóm (80% Train - 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```

> **Lưu ý chống Data Leakage:** Các bước chuẩn hóa (`Scaler`) và xử lý mất cân bằng lớp (`class_weight='balanced'`, `RandomUnderSampler`, `SMOTE`, `TomekLinks`) **chỉ được thực hiện trên `X_train, y_train`** sau bước `train_test_split` ở trên.
