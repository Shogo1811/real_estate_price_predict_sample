import pandas as pd
import random

# サンプルデータの作成
# 既存のデータを元に拡張
data = [
    {"area": 30, "year_built": 2005, "num_rooms": 1, "location": "central", "rent": 85000},
    {"area": 45, "year_built": 1998, "num_rooms": 2, "location": "suburban", "rent": 70000},
    {"area": 50, "year_built": 2010, "num_rooms": 2, "location": "central", "rent": 100000},
    {"area": 40, "year_built": 2000, "num_rooms": 1, "location": "suburban", "rent": 60000},
    {"area": 35, "year_built": 2015, "num_rooms": 1, "location": "rural", "rent": 50000},
]

# 新しいデータをランダム生成
locations = ["central", "suburban", "rural"]
for _ in range(1000):  # 追加するデータ数
    data.append({
        "area": random.randint(20, 100),  # 面積を20～100平米の範囲で生成
        "year_built": random.randint(1980, 2022),  # 築年を1980～2022年で生成
        "num_rooms": random.randint(1, 5),  # 部屋数を1～5で生成
        "location": random.choice(locations),  # ランダムなロケーション
        "rent": random.randint(40000, 150000),  # 家賃を4万円～15万円で生成
    })

# データフレームに変換
df = pd.DataFrame(data)

# CSVファイルに保存
df.to_csv("rent_data_extended.csv", index=False)

print("Extended data saved to 'rent_data_extended.csv'.")
