import pandas as pd
import random

# サンプルデータの作成
# 東京23区の区名
locations = [
    "Chiyoda", "Chuo", "Minato", "Shinjuku", "Bunkyo", "Taito", "Sumida", "Koto",
    "Shinagawa", "Meguro", "Ota", "Setagaya", "Shibuya", "Nakano", "Suginami",
    "Toshima", "Kita", "Arakawa", "Itabashi", "Nerima", "Adachi", "Katsushika", "Edogawa"
]

# ランダムデータの生成
data = []
for _ in range(500):  # 500件のデータを作成
    area = random.randint(20, 100)  # 面積 (20～100平米)
    year_built = random.randint(1980, 2022)  # 築年 (1980～2022)
    num_rooms = random.randint(1, 5)  # 部屋数 (1～5)
    location = random.choice(locations)  # 東京23区のランダムな区
    rent = random.randint(50000, 300000)  # 家賃 (50,000円～300,000円)
    data.append({
        "area": area,
        "year_built": year_built,
        "num_rooms": num_rooms,
        "location": location,
        "rent": rent
    })

# データフレームに変換
df = pd.DataFrame(data)

# CSVファイルとして保存
csv_path = "tokyo_23ku_rent_data.csv"
df.to_csv(csv_path, index=False)

csv_path
