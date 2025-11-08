# ============================================
# GOLD PRICE VISUALIZATION (SJC vs PNJ)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu
file_path = "gold_price.csv"  # đổi đường dẫn nếu cần
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

#  Tính DailyChange
df = df.sort_values(['Brand', 'Date'])
df['DailyChange'] = df.groupby('Brand')['GoldPrice'].diff()
df = df.dropna()

# ============================================
# 1. Bar chart trung bình DailyChange theo tháng
# ============================================
avg_change = df.groupby(['Brand', 'Month'])['DailyChange'].mean().unstack('Brand')

plt.figure(figsize=(10, 6))
avg_change.plot(kind='bar', figsize=(10, 6))
plt.title("Trung bình DailyChange theo tháng (PNJ vs SJC)")
plt.xlabel("Tháng")
plt.ylabel("DailyChange trung bình (nghìn VND)")
plt.legend(title="Thương hiệu")
plt.tight_layout()
plt.show()

# ============================================
# 2. Boxplot DailyChange theo tháng (PNJ left, SJC right)
# ============================================
plt.figure(figsize=(12, 6))
brands = ['PNJ', 'SJC']
months = sorted(df['Month'].unique())
positions = []
data = []

for i, month in enumerate(months):
    for j, brand in enumerate(brands):
        subset = df[(df['Month'] == month) & (df['Brand'] == brand)]['DailyChange']
        data.append(subset)
        positions.append(i * 3 + j + 1)

plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
plt.title("Boxplot DailyChange theo tháng (PNJ left, SJC right)")
plt.xlabel("Tháng")
plt.ylabel("DailyChange (nghìn VND)")
plt.xticks([(i*3)+1.5 for i in range(len(months))], months)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==============
# 3. Line chart 
# ==============
plt.figure(figsize=(12, 6))
for brand in df['Brand'].unique():
    subset = df[df['Brand'] == brand]
    plt.plot(subset['Date'], subset['GoldPrice'], label=brand)

plt.title("Giá vàng thực tế theo thời gian (SJC vs PNJ)")
plt.xlabel("Ngày")
plt.ylabel("Giá vàng (nghìn VND)")
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
