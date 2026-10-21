
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# ==========================================
# BƯỚC 1. ĐỌC DỮ LIỆU ĐÃ CHUẨN HÓA
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

feature_names = df.drop(columns=["target"]).columns

print("\nKích thước dữ liệu:", X_scaled.shape)
print("Số mẫu:", X_scaled.shape[0])
print("Số thuộc tính:", X_scaled.shape[1])


# ==========================================
# BƯỚC 2. TÍNH MA TRẬN HIỆP PHƯƠNG SAI
# ==========================================

cov_matrix = np.cov(X_scaled, rowvar=False)

print("\n=== MA TRẬN HIỆP PHƯƠNG SAI ===")
print("Kích thước:", cov_matrix.shape)
print(np.round(cov_matrix, 4))


# Vẽ biểu đồ ma trận hiệp phương sai
plt.figure(figsize=(10, 8))

plt.imshow(cov_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Hiệp phương sai")

plt.xticks(
    range(len(feature_names)),
    feature_names,
    rotation=90
)

plt.yticks(
    range(len(feature_names)),
    feature_names
)

plt.title("Ma trận hiệp phương sai")
plt.tight_layout()
plt.show()


# ==========================================
# BƯỚC 3. TÍNH TRỊ RIÊNG VÀ VECTOR RIÊNG
# ==========================================

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sắp xếp trị riêng giảm dần
indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:, indices]

print("\n=== TRỊ RIÊNG ===")
print(np.round(eigenvalues, 4))

print("\n=== VECTOR RIÊNG CỦA PC1 ===")
print(np.round(eigenvectors[:, 0], 4))

print("\n=== VECTOR RIÊNG CỦA PC2 ===")
print(np.round(eigenvectors[:, 1], 4))


# ==========================================
# BƯỚC 4. CHỌN THÀNH PHẦN CHÍNH
# ==========================================

# Tính tỷ lệ phương sai giải thích
explained_variance_ratio = (
    eigenvalues / np.sum(eigenvalues)
)

print("\n=== TỶ LỆ PHƯƠNG SAI GIẢI THÍCH ===")

for i in range(len(explained_variance_ratio)):
    print(
        f"PC{i + 1}: "
        f"{explained_variance_ratio[i] * 100:.2f}%"
    )

# Chọn 2 vector riêng ứng với 2 trị riêng lớn nhất
n_components = 2
W = eigenvectors[:, :n_components]

print("\nMa trận thành phần chính W:")
print(np.round(W, 4))
print("Kích thước W:", W.shape)

print(
    "\nPhương sai giữ lại với 2 PC:",
    f"{np.sum(explained_variance_ratio[:2]) * 100:.2f}%"
)


# Vẽ tỷ lệ phương sai giải thích của từng PC
plt.figure(figsize=(9, 5))

plt.bar(
    range(1, len(explained_variance_ratio) + 1),
    explained_variance_ratio * 100
)

plt.xlabel("Thành phần chính")
plt.ylabel("Tỷ lệ phương sai giải thích (%)")
plt.title("Explained Variance Ratio")
plt.xticks(range(1, len(explained_variance_ratio) + 1))

plt.tight_layout()
plt.show()


# ==========================================
# BƯỚC 5. CHIẾU DỮ LIỆU XUỐNG 2 CHIỀU
# ==========================================

# Chiếu dữ liệu 13 chiều xuống 2 chiều
X_pca = X_scaled @ W

print("\n=== KẾT QUẢ SAU PCA ===")
print("Kích thước ban đầu:", X_scaled.shape)
print("Kích thước sau PCA:", X_pca.shape)

# Tạo bảng kết quả
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["target"] = y

print("\n5 mẫu đầu tiên sau PCA:")
print(pca_df.head())


# ==========================================
# BƯỚC 6. VẼ BIỂU ĐỒ PHÂN TÁN PCA
# ==========================================

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
plt.title("PCA - Wine Dataset (Manual)")
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

# Vẽ biểu đồ phương sai tích lũy
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
plt.title("Cumulative Explained Variance")
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================
# BƯỚC 8. LƯU KẾT QUẢ PCA
# ==========================================

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "wine_pca_manual.csv"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pca_df.to_csv(OUTPUT_PATH, index=False)

print("\nĐã lưu kết quả PCA tại:")
print(OUTPUT_PATH)
