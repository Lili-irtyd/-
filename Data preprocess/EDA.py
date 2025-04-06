import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# 異常値を移動平均で置き換える関数
def replace_outliers_with_moving_avg(df, column_name="OT"):
    df = df.copy()
    moving_avg = df[column_name].rolling(window=5, min_periods=1).mean()
    df["moving_avg"] = moving_avg
    z_scores = (df[column_name] - df[column_name].mean()) / df[column_name].std()
    df.loc[z_scores.abs() > 1.5, column_name] = df["moving_avg"]
    df.drop(columns=["moving_avg"], inplace=True)
    return df

# データの読み込み
df = pd.read_csv("業務課題\Dataset\ett.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# 基本統計量の確認
print("基本統計量：")
print(df.describe())

# \ターゲット（油温）だけ抽出
target_col = 'OT'
oil_temp = df[target_col]

# トレンドと季節性（分解）
decomposition = seasonal_decompose(oil_temp, model='additive', period=1440)

plt.figure(figsize=(16, 10))
plt.subplot(411)
plt.plot(decomposition.observed, label='Observed')
plt.title('観測値（油温）')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='orange')
plt.title('トレンド')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal', color='green')
plt.title('季節性（日周期）')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual', color='red')
plt.title('残差')
plt.tight_layout()
plt.show()

# 異常値可視化（置き換え前）
plt.figure(figsize=(8, 6))
sns.boxplot(y=oil_temp)
plt.title("油温のボックスプロット（異常値検出・置換前）")
plt.grid(True)
plt.show()

# 異常値を移動平均で置換
df_cleaned = replace_outliers_with_moving_avg(df, column_name=target_col)

# 清理后油温再画图
plt.figure(figsize=(18, 5))
plt.plot(df_cleaned[target_col], label='Cleaned Oil Temperature (replaced)', color='teal')
plt.title("移動平均で異常値を置換した後の油温時系列")
plt.xlabel("日付")
plt.ylabel("油温")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
