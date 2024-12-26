import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# === 1. CSVファイルを読み込む ===
file_path = "Tokyo_20242_20242.csv"  # CSVファイルのパスを指定
df = pd.read_csv(file_path, sep="\t", encoding="utf-8")  # タブ区切り

# 必要な列を選択
columns_to_use = ["取引価格（総額）", "面積（㎡）", "最寄駅：距離（分）", "建築年", "市区町村名"]
df = df[columns_to_use]

# === 2. データの前処理 ===

# "建築年"を数値化
df["建築年"] = pd.to_numeric(df["建築年"].str.replace("年", ""), errors="coerce")

# "最寄駅：距離（分）"を数値化
def parse_distance(distance):
    if isinstance(distance, str):
        if "H" in distance:  # "1H" や "1H30分"
            parts = distance.replace("分", "").split("H")
            hours = int(parts[0]) * 60
            minutes = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            return hours + minutes
        elif "～" in distance:  # "30分～60分"
            parts = distance.replace("分", "").split("～")
            return (int(parts[0]) + int(parts[1])) / 2
        elif "分" in distance:  # "5分"
            return int(distance.replace("分", ""))
    return distance  # 数値の場合はそのまま返す

df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].apply(parse_distance)

# 欠損値を削除
df = df.dropna()

# 市区町村名をOne-Hot Encoding
df = pd.get_dummies(df, columns=["市区町村名"], drop_first=True)

# 特徴量（X）とターゲット（y）に分割
X = df.drop("取引価格（総額）", axis=1)
y = df["取引価格（総額）"]

# === 3. データ分割 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. モデルの構築とトレーニング ===
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# === 5. テストデータで予測 ===
y_pred = model.predict(X_test)

# === 6. 結果を評価 ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# === 7. 実際の価格と予測価格を比較 ===
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})
print(comparison.head())

# 結果をCSVに保存
output_file = "predicted_results.csv"
comparison.to_csv(output_file, index=False, encoding="utf-8")
print(f"予測結果を '{output_file}' に保存しました。")

# === 8. 実際の価格 vs 予測価格をプロット ===
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.7, label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label="Ideal Prediction")
plt.xlabel("Actual Price (yen)")
plt.ylabel("Predicted Price (yen)")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.grid()
plt.show()
