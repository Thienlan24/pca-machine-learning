import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.pca.data_loader import load_wine_data


# Đường dẫn thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Đường dẫn lưu dữ liệu sau chuẩn hóa
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "wine_standardized.csv"


def standardize_data(X):
    """
    Chuẩn hóa dữ liệu bằng StandardScaler.

    Đầu vào:
        X: Ma trận đặc trưng (178, 13)

    Đầu ra:
        X_scaled: Ma trận đã chuẩn hóa (178, 13)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


if __name__ == "__main__":
    # Bước 1: Đọc dữ liệu Wine
    X, y, feature_names = load_wine_data()

    # Bước 2: Chuẩn hóa 13 đặc trưng
    X_scaled = standardize_data(X)

    # Bước 3: Tạo DataFrame có tên cột
    df_scaled = pd.DataFrame(
        X_scaled,
        columns=feature_names
    )

    # Giữ nguyên nhãn phân lớp
    df_scaled.insert(0, "target", y)

    # Bước 4: Lưu dữ liệu đã chuẩn hóa
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(OUTPUT_PATH, index=False)

    # Bước 5: Kiểm tra kết quả
    print("=== CHUẨN HÓA DỮ LIỆU WINE ===")
    print("Kích thước trước chuẩn hóa:", X.shape)
    print("Kích thước sau chuẩn hóa:", X_scaled.shape)

    print("\nTrung bình của 13 đặc trưng:")
    print(np.round(X_scaled.mean(axis=0), 6))

    print("\nĐộ lệch chuẩn của 13 đặc trưng:")
    print(np.round(X_scaled.std(axis=0), 6))

    print("\n5 mẫu đầu tiên sau chuẩn hóa:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df_scaled.head())

    print("\nĐã lưu file tại:")
    print(OUTPUT_PATH)