import pandas as pd

# ファイルパスを指定
file_path = "Tokyo_20242_20242.csv"

# 区切り文字としてタブ (\t) を指定して読み込む
df = pd.read_csv(file_path, sep="\t", encoding="utf-8")  # エンコーディングが合わない場合は変更

# データの先頭を確認
print(df.head())
