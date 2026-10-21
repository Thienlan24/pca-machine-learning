import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Thêm thư mục gốc dự án vào đường dẫn tìm module
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pca.data_loader import load_wine_data
from pca.preprocessing import standardize_data


# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

MANUAL_SCRIPT = PROJECT_ROOT / "pca" / "pca_manual.py"
SKLEARN_SCRIPT = PROJECT_ROOT / "pca" / "pca_sklearn.py"


# ==========================================
# 1. CHUẨN HÓA DỮ LIỆU
# ==========================================

def prepare_data():
    print("\n=== 1. ĐỌC VÀ CHUẨN HÓA DỮ LIỆU ===")

    X, y, feature_names = load_wine_data()
    X_scaled = standardize_data(X)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(X_scaled, columns=feature_names)
    df.insert(0, "target", y)

    output_path = DATA_DIR / "wine_standardized.csv"
    df.to_csv(output_path, index=False)

    print("Dữ liệu gốc:", X.shape)
    print("Dữ liệu chuẩn hóa:", X_scaled.shape)
    print("Đã lưu:", output_path)

    return X_scaled, y, feature_names


# ==========================================
# 2. CHẠY CÁC CHƯƠNG TRÌNH PCA
# ==========================================

def run_pca_script(script_path):
    print(f"\n=== CHẠY {script_path.name} ===")

    if not script_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {script_path}")

    # Sử dụng backend không mở cửa sổ biểu đồ
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True
    )


# ==========================================
# 3. ĐỌC KẾT QUẢ PCA
# ==========================================

def load_pca_result(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy kết quả: {path}")

    df = pd.read_csv(path)

    required = ["PC1", "PC2", "target"]

    if not all(col in df.columns for col in required):
        raise ValueError(f"File {filename} thiếu cột PCA.")

    return df


# ==========================================
# 4. VẼ BIỂU ĐỒ PCA 2D
# ==========================================

def plot_pca(df, title, filename):
    plt.figure(figsize=(9, 6))

    for label in sorted(df["target"].unique()):
        subset = df[df["target"] == label]

        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=f"Wine class {label}",
            alpha=0.8
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = FIGURE_DIR / filename
    plt.savefig(path, dpi=300)
    plt.close()

    print("Đã lưu biểu đồ:", path)


# ==========================================
# 5. PHÂN TÍCH PHƯƠNG SAI PCA
# ==========================================

def calculate_variance(X_scaled):
    # Tính phương sai của PCA thủ công
    X_centered = X_scaled - X_scaled.mean(axis=0)

    cov_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)

    manual_ratio = eigenvalues / eigenvalues.sum()

    # Tính phương sai bằng sklearn
    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(X_scaled)

    sklearn_ratio = pca.explained_variance_ratio_

    return manual_ratio, sklearn_ratio


# ==========================================
# 6. VẼ BIỂU ĐỒ PHƯƠNG SAI
# ==========================================

def plot_variance(manual_ratio, sklearn_ratio):
    n_components = len(manual_ratio)
    pcs = np.arange(1, n_components + 1)

    # Biểu đồ tỷ lệ phương sai giải thích
    plt.figure(figsize=(10, 5))

    width = 0.35

    plt.bar(
        pcs - width / 2,
        manual_ratio * 100,
        width,
        label="Manual PCA"
    )

    plt.bar(
        pcs + width / 2,
        sklearn_ratio * 100,
        width,
        label="Sklearn PCA"
    )

    plt.xlabel("Thành phần chính")
    plt.ylabel("Phương sai giải thích (%)")
    plt.title("Explained Variance Ratio - Manual vs Sklearn")
    plt.xticks(pcs)
    plt.legend()
    plt.tight_layout()

    path = FIGURE_DIR / "explained_variance_ratio.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Đã lưu biểu đồ:", path)

    # Biểu đồ phương sai tích lũy
    plt.figure(figsize=(10, 5))

    plt.plot(
        pcs,
        np.cumsum(manual_ratio) * 100,
        marker="o",
        label="Manual PCA"
    )

    plt.plot(
        pcs,
        np.cumsum(sklearn_ratio) * 100,
        marker="x",
        linestyle="--",
        label="Sklearn PCA"
    )

    plt.axhline(
        95,
        linestyle=":",
        color="gray",
        label="Mốc 95%"
    )

    plt.xlabel("Số thành phần chính")
    plt.ylabel("Phương sai tích lũy (%)")
    plt.title("Cumulative Explained Variance")
    plt.xticks(pcs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = FIGURE_DIR / "cumulative_explained_variance.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Đã lưu biểu đồ:", path)


# ==========================================
# 7. VẼ MA TRẬN HIỆP PHƯƠNG SAI
# ==========================================

def plot_covariance(X_scaled, feature_names):
    cov_matrix = np.cov(X_scaled, rowvar=False)

    plt.figure(figsize=(11, 9))

    plt.imshow(
        cov_matrix,
        cmap="coolwarm",
        aspect="auto"
    )

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

    path = FIGURE_DIR / "covariance_heatmap.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Đã lưu biểu đồ:", path)


# ==========================================
# 8. VẼ BOXPLOT TRƯỚC VÀ SAU CHUẨN HÓA
# ==========================================

def plot_standardization(X, X_scaled, feature_names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].boxplot(X)
    axes[0].set_title("Trước chuẩn hóa")
    axes[0].set_ylabel("Giá trị")

    axes[1].boxplot(X_scaled)
    axes[1].set_title("Sau chuẩn hóa")
    axes[1].set_ylabel("Giá trị")

    for ax in axes:
        ax.set_xticks(range(1, len(feature_names) + 1))
        ax.set_xticklabels(
            feature_names,
            rotation=90,
            fontsize=8
        )

    plt.tight_layout()

    path = FIGURE_DIR / "standardization_boxplot.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Đã lưu biểu đồ:", path)


# ==========================================
# 9. XUẤT BẢNG SO SÁNH
# ==========================================

def export_comparison(manual_ratio, sklearn_ratio):
    pcs = [
        f"PC{i + 1}"
        for i in range(len(manual_ratio))
    ]

    df = pd.DataFrame({
        "Principal Component": pcs,
        "Manual PCA": manual_ratio,
        "Sklearn PCA": sklearn_ratio,
        "Difference": np.abs(manual_ratio - sklearn_ratio)
    })

    path = TABLE_DIR / "explained_variance.csv"
    df.to_csv(path, index=False)

    print("\n=== BẢNG SO SÁNH PHƯƠNG SAI ===")
    print(df.to_string(index=False))

    print("\nTổng phương sai 2 PC:")
    print(
        f"Manual: {manual_ratio[:2].sum() * 100:.2f}%"
    )
    print(
        f"Sklearn: {sklearn_ratio[:2].sum() * 100:.2f}%"
    )

    print("Đã lưu bảng:", path)


# ==========================================
# 10. CHẠY TOÀN BỘ THỰC NGHIỆM
# ==========================================

def run_experiment():
    print("=== THỰC NGHIỆM PCA TRÊN DATASET WINE ===")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu và chuẩn hóa
    X, y, feature_names = load_wine_data()
    X_scaled, _, _ = prepare_data()

    # Vẽ biểu đồ chuẩn hóa
    plot_standardization(X, X_scaled, feature_names)

    # Chạy PCA thủ công
    run_pca_script(MANUAL_SCRIPT)

    # Chạy PCA sklearn
    run_pca_script(SKLEARN_SCRIPT)

    # Đọc kết quả từ hai phương pháp
    df_manual = load_pca_result("wine_pca_manual.csv")
    df_sklearn = load_pca_result("wine_pca_sklearn.csv")

    # Kiểm tra số mẫu và nhãn có khớp không
    if len(df_manual) != len(df_sklearn):
        raise ValueError("Hai kết quả PCA có số mẫu khác nhau.")

    if not np.array_equal(
        df_manual["target"].to_numpy(),
        df_sklearn["target"].to_numpy()
    ):
        raise ValueError("Nhãn của hai kết quả PCA không khớp.")

    # Vẽ PCA 2D
    plot_pca(
        df_manual,
        "PCA Manual - Wine Dataset",
        "pca_manual_2d.png"
    )

    plot_pca(
        df_sklearn,
        "PCA Sklearn - Wine Dataset",
        "pca_sklearn_2d.png"
    )

    # Phân tích phương sai
    manual_ratio, sklearn_ratio = calculate_variance(X_scaled)

    plot_variance(manual_ratio, sklearn_ratio)
    plot_covariance(X_scaled, feature_names)

    # Xuất bảng so sánh
    export_comparison(manual_ratio, sklearn_ratio)

    if np.allclose(manual_ratio, sklearn_ratio, atol=1e-6):
        print("\nKết luận: Hai phương pháp PCA có phương sai tương đương.")
    else:
        print("\nCảnh báo: Hai phương pháp PCA có sai lệch phương sai.")

    print("\n=== HOÀN THÀNH THỰC NGHIỆM PCA ===")


if __name__ == "__main__":
    run_experiment()
