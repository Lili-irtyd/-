# This script processes oil temperature (OT) data by:
# 1. Handling missing values with forward-fill and mean imputation.
# 2. Standardizing and normalizing the data using StandardScaler and MinMaxScaler.
# 3. Engineering time-related features (year, month, day, hour, weekday) and lag features.
# 4. Calculating rolling mean and standard deviation for oil temperature over a 24-hour window.
# 5. Generating visualizations: 
#    - Oil temperature distribution (histogram with KDE).
#    - Feature correlation heatmap.
#    - Oil temperature trend over time.
# All processed data and visualizations are saved in the "picture" directory.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 确保 picture 目录存在
picture_dir = "picture"
os.makedirs(picture_dir, exist_ok=True)

# 读取数据
df = pd.read_csv("業務課題\Dataset\ett_cleaned.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

### 1. 处理缺失值 ###
# 统计缺失值
missing_values = df.isnull().sum()
missing_values.to_csv(os.path.join(picture_dir, "missing_values.csv"))  # 保存缺失值统计信息

# 填充缺失值（使用前向填充+均值填充）
df.fillna(method="ffill", inplace=True)  # 先用前值填充
df.fillna(df.mean(), inplace=True)  # 再用均值填充

### 2. 数据标准化和归一化 ###
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

df_standardized = pd.DataFrame(scaler_standard.fit_transform(df), columns=df.columns, index=df.index)
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns, index=df.index)

df_standardized.to_csv(os.path.join(picture_dir,"standardized_data.csv"))  # 保存标准化数据
df_minmax.to_csv(os.path.join(picture_dir, "minmax_scaled_data.csv"))  # 保存归一化数据

### 3. 特征工程 ###
# 添加时间特征
df["year"] = df.index.year
df["month"] = df.index.month
df["day"] = df.index.day
df["hour"] = df.index.hour
df["weekday"] = df.index.weekday  # 0-6, 代表周一到周日

# 计算值油温的滚动均和滚动标准差（移动窗口特征）
df["OT_rolling_mean_24h"] = df["OT"].rolling(window=24).mean()
df["OT_rolling_std_24h"] = df["OT"].rolling(window=24).std()

# 计算油温的滞后特征（时序特征）
for lag in [1, 3, 6, 12, 24]:  # 1小时、3小时、6小时、12小时、24小时滞后
    df[f"OT_lag_{lag}"] = df["OT"].shift(lag)

df.dropna(inplace=True)  # 由于滞后特征和滚动计算会产生NaN，去除这些行

df.to_csv(os.path.join(picture_dir, "processed_data.csv"))  # 保存处理后的数据

### 4. 生成数据分布和相关性图 ###
# 油温（OT）分布图
plt.figure(figsize=(8, 4))
sns.histplot(df["OT"], kde=True, bins=50)
plt.title("Distribution of Oil Temperature (OT)")
plt.savefig(os.path.join(picture_dir, "OT_distribution.svg"), format="svg")
plt.close()

# 相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(picture_dir, "correlation_heatmap.svg"), format="svg")
plt.close()

# 油温随时间变化趋势
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["OT"], label="Oil Temperature", color="blue")
plt.xlabel("Date")
plt.ylabel("OT")
plt.title("Oil Temperature Over Time")
plt.legend()
plt.savefig(os.path.join(picture_dir, "OT_trend.svg"), format="svg")
plt.close()

