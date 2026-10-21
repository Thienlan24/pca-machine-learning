import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA


# ==========================================
# BƯỚC 1. ĐỌC DỮ LIỆU ĐÃ CHUẨN HÓA
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "wine_standardized.csv"
)

print("Đường dẫn dữ liệu:", DATA_PATH)
print("File tồn tại:", DATA_PATH.exists())

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Không tìm thấy wine_standardized.csv. "
        "Hãy chạy preprocessing.py trước."
    )

df = pd.read_csv(DATA_PATH)

# Tách nhãn và các thuộc tính đã chuẩn hóa
y = df["target"].to_numpy()
X_scaled = df.drop(columns=["target"]).to_numpy(dtype=float)

print("\nKích thước dữ liệu:", X_scaled.shape)
print("Số mẫu:", X_scaled.shape[0])
print("Số thuộc tính:", X_scaled.shape[1])


# ==========================================
# BƯỚC 2. KHỞI TẠO VÀ HUẤN LUYỆN PCA
# ==========================================

# PCA đầy đủ để phân tích tất cả thành phần chính
pca_full = PCA()

pca_full.fit(X_scaled)

print("\n=== PCA BẰNG SCIKIT-LEARN ===")
print("Số thành phần chính:", pca_full.n_components_)


# ==========================================
# BƯỚC 3. LẤY PHƯƠNG SAI GIẢI THÍCH
# ==========================================

explained_variance_ratio = (
    pca_full.explained_variance_ratio_
)

print("\n=== TỶ LỆ PHƯƠNG SAI GIẢI THÍCH ===")

for i in range(len(explained_variance_ratio)):
    print(
        f"PC{i + 1}: "
        f"{explained_variance_ratio[i] * 100:.2f}%"
    )

print("\n=== TRỊ RIÊNG (EXPLAINED VARIANCE) ===")
print(np.round(pca_full.explained_variance_, 4))


# ==========================================
# BƯỚC 4. CHỌN 2 THÀNH PHẦN CHÍNH
# ==========================================

n_components = 2

pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X_scaled)

print("\n=== KẾT QUẢ SAU PCA ===")
print("Kích thước ban đầu:", X_scaled.shape)
print("Kích thước sau PCA:", X_pca.shape)

print(
    "\nPhương sai giữ lại với 2 PC:",
    f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
)

print("\nCác vector thành phần chính:")
print(np.round(pca.components_, 4))


# ==========================================
# BƯỚC 5. VẼ BIỂU ĐỒ PHƯƠNG SAI GIẢI THÍCH
# ==========================================

plt.figure(figsize=(9, 5))

plt.bar(
    range(1, len(explained_variance_ratio) + 1),
    explained_variance_ratio * 100
)

plt.xlabel("Thành phần chính")
plt.ylabel("Tỷ lệ phương sai giải thích (%)")
plt.title("Explained Variance Ratio - Sklearn PCA")
plt.xticks(range(1, len(explained_variance_ratio) + 1))

plt.tight_layout()
plt.show()


# ==========================================
# BƯỚC 6. VẼ BIỂU ĐỒ PHÂN TÁN PCA 2D
# ==========================================

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["target"] = y

plt.figure(figsize=(9, 6))

for label in sorted(np.unique(y)):
    mask = y == label

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Wine class {label}",
        alpha=0.8
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Wine Dataset (Sklearn)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# BƯỚC 7. PHƯƠNG SAI GIẢI THÍCH TÍCH LŨY
# ==========================================

cumulative_variance = np.cumsum(explained_variance_ratio)

print("\n=== PHƯƠNG SAI GIẢI THÍCH TÍCH LŨY ===")

for i in range(len(cumulative_variance)):
    print(
        f"PC1 đến PC{i + 1}: "
        f"{cumulative_variance[i] * 100:.2f}%"
    )

print(
    "\nPhương sai giữ lại với 2 PC:",
    f"{cumulative_variance[1] * 100:.2f}%"
)

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance * 100,
    marker="o"
)

plt.axhline(
    y=95,
    linestyle="--",
    label="Mốc 95%"
)

plt.xlabel("Số thành phần chính")
plt.ylabel("Phương sai tích lũy (%)")
plt.title("Cumulative Explained Variance - Sklearn PCA")
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================
# BƯỚC 8. LƯU KẾT QUẢ PCA
# ==========================================

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "wine_pca_sklearn.csv"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pca_df.to_csv(OUTPUT_PATH, index=False)

print("\nĐã lưu kết quả PCA tại:")
print(OUTPUT_PATH)
