# ============================================
# GOLD PRICE ANALYSIS (SJC vs PNJ)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1️⃣ Đọc dữ liệu
file_path = "gold_price.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Brand', 'Date'])
df['Month'] = df['Date'].dt.month

# 2️⃣ Tính thay đổi hàng ngày
df['DailyChange'] = df.groupby('Brand')['GoldPrice'].diff()
df = df.dropna()

# ==========================================================
# 📘 CÂU 2 — One-sample t-test (μ > 80 ?)
# ==========================================================
print("\n===================== CÂU 2 — Mean of population =====================")
print("""
 Kiểm định giả thuyết + khoảng tin cậy cho trung bình (1 tổng thể)
Mục tiêu: giá vàng trung bình 9 tháng đầu có lớn hơn 80 không (tức là có tăng giá thật sự không).
Giả thuyết:
    H₀: μ = 80 (trung bình giá vàng = 80 ⇒ giá không tăng)
    H₁: μ > 80 (trung bình giá vàng > 80 ⇒ giá tăng)
→ Dùng one-sample t-test
→ Sau đó xây dựng khoảng tin cậy 95% cho trung bình GoldPrice.
""")

for brand in df['Brand'].unique():
    subset = df[df['Brand'] == brand]['GoldPrice']
    n = len(subset)
    mean = subset.mean()
    std = subset.std(ddof=1)
    mu0 = 80
    t_stat, p_val_two = stats.ttest_1samp(subset, mu0)
    p_val_one = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2
    conf_int = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
    
    print(f"\n🔹 {brand}:")
    print(f"n = {n}, Mean = {mean:.6f}, Std = {std:.6f}")
    print(f"t = {t_stat:.4f}, p (one-sided) = {p_val_one:.6f}")
    print(f"95% CI for μ: {conf_int}")
    if p_val_one < 0.05:
        print("➡️ Kết luận: Bác bỏ H₀ → Giá vàng trung bình > 80 (có xu hướng tăng).")
    else:
        print("➡️ Kết luận: Không bác bỏ H₀ → Không đủ bằng chứng giá vàng trung bình > 80.")

# ==========================================================
# 📗 CÂU 3 — One-proportion z-test (p > 0.5 ?)
# ==========================================================
print("\n===================== CÂU 3 —  proportion of a population =====================")
print("""
Kiểm định giả thuyết + khoảng tin cậy cho tỷ lệ (1 tổng thể)
Mục tiêu: Xem tỷ lệ số ngày giá tăng có vượt 50% không.
Cách làm:
    - Đếm số ngày có DailyChange > 0 → “ngày tăng giá”
    - Tính tỷ lệ p = (số ngày tăng / tổng số ngày)
Giả thuyết:
    H₀: p = 0.5
    H₁: p > 0.5
→ Dùng one-proportion z-test
→ Xây dựng khoảng tin cậy 95% cho p.
""")

for brand in df['Brand'].unique():
    subset = df[df['Brand'] == brand]['DailyChange']
    n = len(subset)
    n_up = np.sum(subset > 0)
    p_hat = n_up / n
    p0 = 0.5
    z = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
    p_val_one = 1 - stats.norm.cdf(z)
    conf_int = stats.norm.interval(0.95, loc=p_hat, scale=np.sqrt(p_hat*(1-p_hat)/n))
    
    print(f"\n🔹 {brand}:")
    print(f"n = {n}, p̂ = {p_hat:.6f}, z = {z:.4f}, p (one-sided) = {p_val_one:.6f}")
    print(f"95% CI for p: {conf_int}")
    if p_val_one < 0.05:
        print("➡️ Kết luận: Bác bỏ H₀ → Tỷ lệ ngày tăng > 50%.")
    else:
        print("➡️ Kết luận: Không bác bỏ H₀ → Không đủ bằng chứng tỷ lệ tăng > 50%.")

# ==========================================================
# 📙 CÂU 4 — Two-sample t-test (so sánh tháng 3 vs tháng 8)
# ==========================================================
print("\n===================== CÂU 4 — difference in means of 2 populations =====================")
print("""
 Kiểm định và khoảng tin cậy cho sự khác biệt trung bình giữa 2 tháng
Mục tiêu: So sánh mức tăng trung bình giữa hai tháng, ví dụ tháng 3 và tháng 8.
Giả thuyết:
    H₀: μ₁ = μ₂ (mức tăng trung bình tháng 3 = tháng 8)
    H₁: μ₁ ≠ μ₂
→ Dùng two-sample t-test (equal variances)
→ Xây dựng khoảng tin cậy 95% cho (μ₁ − μ₂)
Kết quả:
    Nếu H₀ bị bác bỏ → Tháng 8 có mức tăng khác đáng kể so với tháng 3.
""")

month1, month2 = 3, 8
for brand in df['Brand'].unique():
    data1 = df[(df['Brand'] == brand) & (df['Month'] == month1)]['DailyChange']
    data2 = df[(df['Brand'] == brand) & (df['Month'] == month2)]['DailyChange']
    
    if len(data1) == 0 or len(data2) == 0:
        print(f"\n⚠️ {brand}: Không có đủ dữ liệu cho tháng {month1} và {month2}.")
        continue

    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=True)
    mean_diff = data1.mean() - data2.mean()
    sp = np.sqrt(((len(data1)-1)*data1.var() + (len(data2)-1)*data2.var()) / (len(data1)+len(data2)-2))
    conf_int = (
        mean_diff - stats.t.ppf(0.975, len(data1)+len(data2)-2)*sp*np.sqrt(1/len(data1)+1/len(data2)),
        mean_diff + stats.t.ppf(0.975, len(data1)+len(data2)-2)*sp*np.sqrt(1/len(data1)+1/len(data2))
    )
    
    print(f"\n🔹 {brand}:")
    print(f"Mean(Tháng {month1}) = {data1.mean():.6f}, Mean(Tháng {month2}) = {data2.mean():.6f}")
    print(f"t = {t_stat:.4f}, p (two-sided) = {p_val:.6f}")
    print(f"95% CI for diff: {conf_int}")
    if p_val < 0.05:
        print("➡️ Kết luận: Bác bỏ H₀ → Có khác biệt giữa hai tháng.")
    else:
        print("➡️ Kết luận: Không bác bỏ H₀ → Không có khác biệt rõ rệt.")

# ==========================================================
# 📕 CÂU 5 — Two-proportion z-test (so sánh tỉ lệ ngày tăng giữa 2 tháng)
# ==========================================================
print("\n===================== CÂU 5 — difference in proportions of 2 populationst =====================")
print("""
Kiểm định và khoảng tin cậy cho sự khác biệt tỷ lệ giữa 2 tháng
Mục tiêu: So sánh tỷ lệ ngày tăng giá giữa hai tháng (tháng 3 và tháng 8).
Cách làm:
    - Tính p₁ = tỷ lệ ngày tăng giá tháng 3
    - Tính p₂ = tỷ lệ ngày tăng giá tháng 8
Giả thuyết:
    H₀: p₁ = p₂
    H₁: p₁ ≠ p₂
→ Dùng two-proportion z-test
→ Xây dựng khoảng tin cậy 95% cho (p₁ − p₂)
""")

for brand in df['Brand'].unique():
    data1 = df[(df['Brand'] == brand) & (df['Month'] == month1)]['DailyChange']
    data2 = df[(df['Brand'] == brand) & (df['Month'] == month2)]['DailyChange']
    if len(data1) == 0 or len(data2) == 0:
        print(f"\n⚠️ {brand}: Không có đủ dữ liệu cho tháng {month1} và {month2}.")
        continue
    
    n1, n2 = len(data1), len(data2)
    p1, p2 = np.mean(data1 > 0), np.mean(data2 > 0)
    p_pool = (p1*n1 + p2*n2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    p_val = 2*(1 - stats.norm.cdf(abs(z)))
    conf_int = (
        (p1 - p2) - 1.96*np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2)),
        (p1 - p2) + 1.96*np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    )
    
    print(f"\n🔹 {brand}:")
    print(f"p1 (Tháng {month1}) = {p1:.6f}, p2 (Tháng {month2}) = {p2:.6f}")
    print(f"z = {z:.4f}, p (two-sided) = {p_val:.6f}")
    print(f"95% CI for p1 - p2: {conf_int}")
    if p_val < 0.05:
        print("➡️ Kết luận: Bác bỏ H₀ → Có khác biệt tỷ lệ ngày tăng giữa 2 tháng.")
    else:
        print("➡️ Kết luận: Không bác bỏ H₀ → Không có khác biệt rõ rệt.")
