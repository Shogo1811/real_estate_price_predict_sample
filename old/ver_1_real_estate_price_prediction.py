import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# データの読み込み
data = pd.read_csv("rent_data.csv")

# 特徴量とターゲット（目的変数）の分割
X = data[["area", "year_built", "num_rooms", "location"]]
y = data["rent"]

# カテゴリ変数（location）のエンコーディング
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
location_encoded = encoder.fit_transform(X[["location"]])

# エンコード結果に列名を付与
location_encoded_df = pd.DataFrame(
    location_encoded,
    columns=encoder.get_feature_names_out(["location"])
)

# 数値データとエンコード結果の結合
X = pd.concat([X.drop("location", axis=1).reset_index(drop=True), location_encoded_df], axis=1)
X.columns = X.columns.astype(str)

# データをトレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 線形回帰モデルの学習
model = LinearRegression()
model.fit(X_train, y_train)

# テストデータを使って予測
y_pred = model.predict(X_test)

# モデルの評価（平均二乗誤差）
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 予測結果の可視化
plt.figure(figsize=(8, 6))

# 実際の値と予測値の散布図を作成
plt.scatter(y_test.values, y_pred, alpha=0.7, label="Predicted values", color="blue")

# 実際の値と予測値が一致する補助線を追加
max_value = max(y_test.max(), y_pred.max())
min_value = min(y_test.min(), y_pred.min())
plt.plot([min_value, max_value], [min_value, max_value], 'r--', label="Perfect prediction", alpha=0.7)

# 各予測値と実際値の間の誤差を線で可視化
for actual, predicted in zip(y_test.values, y_pred):
    plt.plot([actual, actual], [actual, predicted], 'gray', alpha=0.5)

# グラフの装飾（軸ラベル、タイトル、凡例、グリッド）
plt.xlabel("Actual Rent (JPY)", fontsize=12)
plt.ylabel("Predicted Rent (JPY)", fontsize=12)
plt.title("Comparison of Actual and Predicted Rent", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# グラフを表示
plt.show()
