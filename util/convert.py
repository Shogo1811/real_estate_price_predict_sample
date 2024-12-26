# csvファイルの文字コード変換
import pandas as pd

# Shift_JISでCSVを読み込む
# df = pd.read_csv("Tokyo_20242_20242.csv", encoding="shift_jis")
df = pd.read_csv("Tokyo_20242_20242.csv", encoding="utf-8")

# UTF-8で保存する
df.to_csv("tokyo_output.csv", index=False, encoding="utf-8")
