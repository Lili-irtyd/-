import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# 读取预处理后的数据
df = pd.read_csv("業務課題\Dataset\cleaned_with_features.csv", parse_dates=["date"])

#**2. 添加时间周期特征**
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df.dropna(inplace=True)
# **保存更新后的数据**
df.to_csv("業務課題\Dataset\modified_processed_data1.csv", index=False)
df=pd.read_csv("業務課題\Dataset\modified_processed_data1.csv",parse_dates=["date"])
# 设定特征和目标变量
target = "OT"
features = [col for col in df.columns if col not in ["date", target]]
X = df[features]
y = df[target]

# 数据拆分 (80% 训练 + 验证, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 进一步划分验证集 (80% 训练, 20% 验证)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 训练多个模型
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Lasso Regression": Lasso(alpha=0.1),
    "Ridge Regression": Ridge(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Support Vector Regression": SVR(kernel='rbf', C=1, epsilon=0.1),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42),
    "CatBoost": CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
}

# 记录模型表现
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred)
    results[name] = {"MAE": mae, "RMSE": rmse}
    print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# 选择最佳模型进行测试集评估
best_model_name = min(results, key=lambda k: results[k]["RMSE"])
best_model = models[best_model_name]
y_test_pred = best_model.predict(X_test_scaled)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
print(f"Best Model: {best_model_name}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")

# 预测值 vs 真实值可视化
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual", linestyle="dashed")
plt.plot(y_test_pred, label="Predicted")
plt.legend()
plt.title(f"{best_model_name} Prediction vs Actual")
plt.savefig(os.path.join("D:\python\業務課題\picture", "prediction_vs_actual.svg"), format="svg")
plt.close()
