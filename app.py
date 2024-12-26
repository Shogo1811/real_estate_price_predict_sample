import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# === 1. CSVファイルを読み込む ===
file_path = "input/Tokyo_20242_20242.csv"  # CSVファイルのパスを指定
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
y_pred = np.round(y_pred, 2)

# === 6. 結果を評価 ===
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# === 7. 実際の価格と予測価格を比較 ===
comparison = pd.DataFrame({
    "Actual Price": y_test.values,  # y_test.values で値を取得
    "Predicted Price": y_pred
})

# Predicted Price を整数に揃える
comparison["Predicted Price"] = comparison["Predicted Price"].astype(int)

# print(comparison.head())

# # === 8. 実際の価格 vs 予測価格をプロット ===
# # 散布図を描画
plt.figure(figsize=(8, 8))
plt.scatter(comparison["Actual Price"], comparison["Predicted Price"], alpha=0.7, label="Predicted vs Actual")


print(comparison["Actual Price"].head())
print(comparison["Predicted Price"].head())

filtered_comparison = comparison[
    (comparison["Actual Price"] < 100000000) & (comparison["Predicted Price"] < 100000000)
]

# グラフ作成
plt.scatter(filtered_comparison["Actual Price"], filtered_comparison["Predicted Price"], alpha=0.7)
plt.xscale("log")
plt.yscale("log")

# X軸とY軸の範囲をデータに合わせて設定
x_min, x_max = filtered_comparison["Actual Price"].min(), filtered_comparison["Actual Price"].max()
y_min, y_max = filtered_comparison["Predicted Price"].min(), filtered_comparison["Predicted Price"].max()
plt.xlim(x_min * 0.8, x_max * 1.2)  # X軸範囲（データより少し広げる）
plt.ylim(y_min * 0.8, y_max * 1.2)  # Y軸範囲

# 目盛りをデータ範囲内に限定し、フォーマットをカスタマイズ
ax = plt.gca()
xticks = [x for x in [1e6, 5e6, 1e7, 5e7, 1e8] if x_min * 0.8 <= x <= x_max * 1.2]
yticks = [y for y in [1e6, 5e6, 1e7, 5e7, 1e8] if y_min * 0.8 <= y <= y_max * 1.2]
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))  # カンマ区切り
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y):,}'))  # カンマ区切り

plt.plot(
    [comparison["Actual Price"].min(), comparison["Actual Price"].max()],
    [comparison["Actual Price"].min(), comparison["Actual Price"].max()],
    '--r', linewidth=2, label="Ideal Prediction"
)

# 軸ラベルとタイトル
plt.xlabel("Actual Price (log scale)")
plt.ylabel("Predicted Price (log scale)")
plt.title("Zoomed Comparison of Actual vs Predicted Prices (Log Scale)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # グリッドを追加
plt.show()
