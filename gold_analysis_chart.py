# ============================================
# GOLD PRICE ANALYSIS (SJC vs PNJ)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1️⃣ Đọc dữ liệu
df = pd.read_csv("gold_price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# 2️⃣ Tính mức thay đổi hàng ngày theo từng thương hiệu
df = df.sort_values(['Brand', 'Date'])
df['DailyChange'] = df.groupby('Brand')['GoldPrice'].diff()
df = df.dropna()

# --------------------------------------------
# 1. Thống kê mô tả
# --------------------------------------------
summary = df.groupby(['Brand', 'Month'])['DailyChange'].describe()
print("📊 Thống kê mô tả theo tháng và thương hiệu:")
print(summary)

# --------------------------------------------
# 2. Bar chart trung bình mức thay đổi giá vàng theo tháng (SJC vs PNJ)
# --------------------------------------------
plt.figure(figsize=(10, 6))
avg_change = df.groupby(['Brand', 'Month'])['DailyChange'].mean().unstack('Brand')
avg_change.plot(kind='bar', figsize=(10, 6))
plt.title("📊 Trung bình mức thay đổi giá vàng theo tháng (SJC vs PNJ)")
plt.xlabel("Tháng")
plt.ylabel("Mức thay đổi trung bình (nghìn VND)")
plt.legend(title="Thương hiệu")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 3. Boxplot DailyChange theo tháng (PNJ left, SJC right per month)
# --------------------------------------------
plt.figure(figsize=(12, 6))
brands = ['PNJ', 'SJC']
months = sorted(df['Month'].unique())
positions = []
data = []
labels = []

# Xây dựng dữ liệu cho boxplot
for i, month in enumerate(months):
    for j, brand in enumerate(brands):
        subset = df[(df['Month'] == month) & (df['Brand'] == brand)]['DailyChange']
        data.append(subset)
        positions.append(i * 3 + j + 1)
        labels.append(f"{brand}-{month}")

plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
plt.title("📦 Boxplot DailyChange theo tháng (PNJ left, SJC right)")
plt.xlabel("Tháng")
plt.ylabel("DailyChange (nghìn VND)")
plt.xticks([(i*3)+1.5 for i in range(len(months))], months)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --------------------------------------------
# 4. Kiểm định trung bình thay đổi ≠ 0 cho từng thương hiệu
# --------------------------------------------
for brand in df['Brand'].unique():
    subset = df[df['Brand'] == brand]['DailyChange']
    t_stat, p_value = stats.ttest_1samp(subset, 0)
    mean = np.mean(subset)
    std = np.std(subset, ddof=1)
    n = len(subset)
    conf_int = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
    print(f"\n🔹 {brand}: One-sample t-test")
    print(f"Mean = {mean:.4f}, t = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"95% CI for mean: {conf_int}")

# --------------------------------------------
# 5. So sánh trung bình giữa SJC và PNJ
# --------------------------------------------
sjc = df[df['Brand'] == 'SJC']['DailyChange']
pnj = df[df['Brand'] == 'PNJ']['DailyChange']

t_stat2, p_val2 = stats.ttest_ind(sjc, pnj, equal_var=True)
mean_diff = sjc.mean() - pnj.mean()
sp = np.sqrt(((len(sjc)-1)*sjc.var() + (len(pnj)-1)*pnj.var()) / (len(sjc)+len(pnj)-2))
conf_int_diff = (
    mean_diff - stats.t.ppf(0.975, len(sjc)+len(pnj)-2)*sp*np.sqrt(1/len(sjc)+1/len(pnj)),
    mean_diff + stats.t.ppf(0.975, len(sjc)+len(pnj)-2)*sp*np.sqrt(1/len(sjc)+1/len(pnj))
)

print("\n⚖️ Two-sample t-test (SJC vs PNJ):")
print(f"Mean diff = {mean_diff:.4f}, t = {t_stat2:.4f}, p = {p_val2:.4f}")
print(f"95% CI for diff: {conf_int_diff}")

# --------------------------------------------
# 6. So sánh tỷ lệ ngày tăng giá giữa 2 thương hiệu
# --------------------------------------------
p1 = np.mean(sjc > 0)
p2 = np.mean(pnj > 0)
n1, n2 = len(sjc), len(pnj)
p_pool = (p1*n1 + p2*n2) / (n1 + n2)
z_diff = (p1 - p2) / np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
p_val_diff = 2*(1 - stats.norm.cdf(abs(z_diff)))
conf_int_pdiff = (
    (p1 - p2) - 1.96*np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2)),
    (p1 - p2) + 1.96*np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
)

print("\n📈 Two-proportion z-test (Tỷ lệ ngày tăng giá SJC vs PNJ):")
print(f"p1 = {p1:.3f}, p2 = {p2:.3f}, z = {z_diff:.4f}, p = {p_val_diff:.4f}")
print(f"95% CI for (p1 - p2): {conf_int_pdiff}")
