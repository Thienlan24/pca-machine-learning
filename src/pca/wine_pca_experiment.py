import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Bước 1. Đọc dữ liệu và tách thuộc tính

# Đọc dữ liệu
# Lấy đường dẫn thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parents[2]

# Đường dẫn đến file dữ liệu
file_path = BASE_DIR / "data" / "raw" / "wine.csv"

print("Đường dẫn file:", file_path)
print("File tồn tại:", file_path.exists())

df = pd.read_csv(file_path)

print("Kích thước dữ liệu:", df.shape)
print("\n5 dòng đầu tiên:")
print(df.head())

# Tách nhãn và các thuộc tính
y = df["target"]
X = df.drop(columns=["target"])

print("\nSố thuộc tính:", X.shape[1])
print("Số mẫu:", X.shape[0])

# Bước 2. Chuẩn hóa dữ liệu

X_array = X.to_numpy(dtype=float)
mean = np.mean(X_array, axis=0)
std = np.std(X_array, axis=0)
X_scaled = (X_array - mean) / std

print("Trung bình từng thuộc tính:")
print(np.round(mean, 4))

print("\nĐộ lệch chuẩn từng thuộc tính:")
print(np.round(std, 4))

print("\nDữ liệu sau chuẩn hóa:")
print(np.round(X_scaled[:5, :5], 4))
print(pd.DataFrame(
    X_scaled,
    columns=X.columns
).iloc[:5, :5])

# Minh họa phân phối trước và sau chuẩn hóa
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.boxplot(X.iloc[:, :5].values)
plt.xticks(range(1, 6), X.columns[:5], rotation=45)
plt.title("Trước chuẩn hóa")
plt.ylabel("Giá trị")

plt.subplot(1, 2, 2)
plt.boxplot(X_scaled[:, :5])
plt.xticks(range(1, 6), X.columns[:5], rotation=45)
plt.title("Sau chuẩn hóa")
plt.ylabel("Giá trị chuẩn hóa")

plt.tight_layout()
plt.show()

# Bước 3. Tính ma trận hiệp phương sai

# Tính ma trận hiệp phương sai
cov_matrix = np.cov(X_scaled, rowvar=False)

print("Kích thước ma trận:", cov_matrix.shape)
print("\nMa trận hiệp phương sai:")
print(np.round(cov_matrix, 2))

# Vẽ heatmap bằng Matplotlib
plt.figure(figsize=(10, 8))

plt.imshow(cov_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Hiệp phương sai")
plt.xticks(range(len(X.columns)), X.columns, rotation=90)
plt.yticks(range(len(X.columns)), X.columns)
plt.title("Ma trận hiệp phương sai")

plt.tight_layout()
plt.show()

# Bước 4. Tính eigenvalue và eigenvector

# Tìm eigenvalue và eigenvector
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sắp xếp theo eigenvalue giảm dần
indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:, indices]

print("Các eigenvalue:")
print(np.round(eigenvalues, 4))

print("\nEigenvector của thành phần chính thứ nhất:")
print(np.round(eigenvectors[:, 0], 4))

print("\nEigenvector của thành phần chính thứ hai:")
print(np.round(eigenvectors[:, 1], 4))

# Tỷ lệ phương sai giải thích
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

print("\nTỷ lệ phương sai giải thích:")
print(np.round(explained_variance_ratio, 4))
# Vẽ tỷ lệ phương sai giải thích của từng thành phần
plt.figure(figsize=(9, 5))

plt.bar(
    range(1, len(eigenvalues) + 1),
    explained_variance_ratio
)

plt.xlabel("Thành phần chính")
plt.ylabel("Tỷ lệ phương sai giải thích")
plt.title("Explained Variance Ratio")
plt.xticks(range(1, len(eigenvalues) + 1))
plt.show()

# Bước 5. Chiếu dữ liệu xuống 2 chiều

# Lấy hai eigenvector đầu tiên
W = eigenvectors[:, :2]

# Chiếu dữ liệu lên PC1 và PC2
X_pca = X_scaled @ W

# Tạo DataFrame kết quả
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["target"] = y.to_numpy()

print(pca_df.head())
print("\nKích thước dữ liệu sau PCA:", X_pca.shape)

# Bước 6. Vẽ biểu đồ PCA

plt.figure(figsize=(9, 6))

for label in sorted(y.unique()):
    mask = y.to_numpy() == label

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Wine class {label}",
        alpha=0.8
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Wine Dataset")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Bước 7. Tính lượng thông tin được giữ lại

# Phương sai giải thích tích lũy
cumulative_variance = np.cumsum(explained_variance_ratio)

for i in range(len(cumulative_variance)):
    print(
        f"PC1 đến PC{i + 1}: "
        f"{cumulative_variance[i] * 100:.2f}%"
    )

# Lượng phương sai được giữ lại khi chọn 2 thành phần
print(
    "\nPhương sai được giữ lại với 2 thành phần:",
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
plt.title("Cumulative Explained Variance")
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()