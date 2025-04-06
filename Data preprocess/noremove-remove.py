import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 读取数据
df = pd.read_csv(r"D:\桌面\業務体験課題\assignment-main\AI-Engineer\multivariate-time-series-prediction\ett.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# 备份原始数据
df_original = df.copy()

# -----------------------------
# 异常值处理：替换 Volume 中 Z-score > 1.5 的值为滚动均值
def replace_outliers_with_moving_avg(df, column_name="OT"):
    df = df.copy()
    moving_avg = df[column_name].rolling(window=5, min_periods=1).mean()
    df["moving_avg"] = moving_avg
    z_scores = (df[column_name] - df[column_name].mean()) / df[column_name].std()
    df.loc[z_scores.abs() > 1.5, column_name] = df["moving_avg"]
    df.drop(columns=["moving_avg"], inplace=True)
    return df

# -----------------------------
# 特征工程函数（保持时间顺序完整）
def apply_feature_engineering(df):
    df_fe = df.copy()  # 保持原始时间顺序
    df_fe["year"] = df_fe.index.year
    df_fe["month"] = df_fe.index.month
    df_fe["day"] = df_fe.index.day
    df_fe["hour"] = df_fe.index.hour
    df_fe["weekday"] = df_fe.index.weekday
    df_fe["OT_rolling_mean_24h"] = df_fe["OT"].rolling(window=24).mean()
    df_fe["OT_rolling_std_24h"] = df_fe["OT"].rolling(window=24).std()
    for lag in [1, 3, 6, 12, 24]:
        df_fe[f"OT_lag_{lag}"] = df_fe["OT"].shift(lag)
    df_fe.dropna(inplace=True)
    return df_fe

# -----------------------------
# 原始数据特征
X_original = df_original.drop("OT", axis=1)
y_original = df_original["OT"]

# -----------------------------
# 替换异常值
df_replaced = replace_outliers_with_moving_avg(df_original, column_name="OT")
X_cleaned = df_replaced.drop("OT", axis=1)
y_cleaned = df_replaced["OT"]

# -----------------------------
# 原始数据 + 特征工程
df_fe = apply_feature_engineering(df_original)
X_fe = df_fe.drop("OT", axis=1)
y_fe = df_fe["OT"]

# 替换异常值 + 特征工程
df_replaced_fe = replace_outliers_with_moving_avg(df_original, column_name="OT")
df_replaced_fe = apply_feature_engineering(df_replaced_fe)
X_cleaned_fe = df_replaced_fe.drop("OT", axis=1)
y_cleaned_fe = df_replaced_fe["OT"]

# 保存到 CSV 文件，确保索引为日期
df_replaced_fe.to_csv("cleaned_with_features.csv", index=True, index_label='date')

# -----------------------------
# 训练与评估函数
def train_and_evaluate(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# -----------------------------
# 实验四种情况
print("\n[1] 原始数据:")
train_and_evaluate(X_original, y_original, "Original")

print("\n[2] 替换异常值:")
train_and_evaluate(X_cleaned, y_cleaned, "Replaced Outliers")

print("\n[3] 原始数据 + 特征工程:")
train_and_evaluate(X_fe, y_fe, "Original + Feature Engineering")

print("\n[4] 替换异常值 + 特征工程:")
train_and_evaluate(X_cleaned_fe, y_cleaned_fe, "Replaced Outliers + Feature Engineering")
