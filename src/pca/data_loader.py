
import pandas as pd
from pathlib import Path

# Tên 13 đặc trưng của bộ dữ liệu Wine
FEATURE_NAMES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline"
]

# Đường dẫn đến thư mục gốc project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Đường dẫn đến dataset CSV có tên cột
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "wine.csv"


def load_wine_data(file_path=DATA_PATH):
    """
    Đọc bộ dữ liệu Wine từ file CSV.

    Returns:
        X: Ma trận 13 đặc trưng, kích thước (178, 13)
        y: Nhãn phân lớp, kích thước (178,)
        feature_names: Danh sách tên 13 đặc trưng
    """

    # CSV đã có dòng tiêu đề nên không dùng header=None
    df = pd.read_csv(file_path)

    # Kiểm tra các cột cần thiết
    required_columns = ["target"] + FEATURE_NAMES
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset thiếu các cột: {missing_columns}"
        )

    # Kiểm tra dữ liệu thiếu
    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset chứa giá trị bị thiếu!")

    # Tách đặc trưng và nhãn
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=int)

    return X, y, FEATURE_NAMES


if __name__ == "__main__":
    X, y, feature_names = load_wine_data()

    print("=== DATASET WINE - UCI ===")
    print("Kích thước X:", X.shape)
    print("Kích thước y:", y.shape)
    print("Số đặc trưng:", len(feature_names))
    print("Các lớp:", sorted(set(y)))

    # Hiển thị đầy đủ tên các cột
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # Tạo bảng để hiển thị tên cột phía trên các giá trị
    df_display = pd.DataFrame(X, columns=feature_names)
    df_display.insert(0, "target", y)

    print("\n5 mẫu đầu tiên (có tên cột):")
    print(df_display.head())

    print("\nSố lượng mẫu theo từng lớp:")
    print(df_display["target"].value_counts().sort_index())
