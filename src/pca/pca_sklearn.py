
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

import sys
from pathlib import Path

# Xác định thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Thêm thư mục gốc vào đường dẫn tìm kiếm module
sys.path.insert(0, str(PROJECT_ROOT))

from src.pca.data_loader import load_wine_data
from src.pca.preprocessing import standardize_data


# Bước 1. Xác định thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Đường dẫn lưu kết quả PCA
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "wine_pca_sklearn.csv"


if __name__ == "__main__":

    # Bước 2. Đọc dữ liệu Wine
    X, y, feature_names = load_wine_data()

    print("Kích thước dữ liệu ban đầu:", X.shape)

    # Bước 3. Chuẩn hóa dữ liệu
    X_scaled = standardize_data(X)

    print("Kích thước dữ liệu sau chuẩn hóa:", X_scaled.shape)

    # Bước 4. Thực hiện PCA bằng Scikit-learn
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    # Bước 5. Lấy tỷ lệ phương sai giải thích
    explained_variance_ratio = pca.explained_variance_ratio_

    cumulative_variance = np.sum(explained_variance_ratio)

    print("\n=== PCA BẰNG SCIKIT-LEARN ===")

    print("\nKích thước dữ liệu sau PCA:", X_pca.shape)

    print("\nTỷ lệ phương sai giải thích:")
    print("PC1:", round(explained_variance_ratio[0] * 100, 2), "%")
    print("PC2:", round(explained_variance_ratio[1] * 100, 2), "%")

    print(
        "Tổng phương sai giữ lại:",
        round(cumulative_variance * 100, 2),
        "%"
    )

    # Bước 6. Tạo DataFrame kết quả
    pca_df = pd.DataFrame(
        X_pca,
        columns=["PC1", "PC2"]
    )

    pca_df.insert(0, "target", np.asarray(y))

    # Tạo thư mục nếu chưa tồn tại
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Lưu kết quả
    pca_df.to_csv(OUTPUT_PATH, index=False)

    print("\n5 mẫu đầu tiên sau PCA:")
    print(pca_df.head())

    print("\nĐã lưu kết quả tại:")
    print(OUTPUT_PATH)

    # Bước 7. Vẽ biểu đồ PCA
    plt.figure(figsize=(9, 6))

    for label in sorted(np.unique(y)):
        mask = np.asarray(y) == label

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=f"Wine class {label}",
            alpha=0.8
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA - Wine Dataset (Scikit-learn)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Bước 8. Vẽ tỷ lệ phương sai giải thích
    plt.figure(figsize=(8, 5))

    plt.bar(
        ["PC1", "PC2"],
        explained_variance_ratio * 100
    )

    plt.ylabel("Tỷ lệ phương sai giải thích (%)")
    plt.title("Explained Variance Ratio - Scikit-learn")
    plt.tight_layout()
    plt.show()
