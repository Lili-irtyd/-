import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# 4. 特征和目标变量定义
df = pd.read_csv("業務課題\Dataset\ett_cleaned.csv", parse_dates=["date"])
target = "OT"
features = [col for col in df.columns if col not in ["date", target]]
X = df[features]
y = df[target]

# 5. 数据拆分：训练+验证 和 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 6. 特征标准化
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# 7. 定义 LightGBM 模型和参数网格
model = lgb.LGBMRegressor(random_state=42)
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 80, 100, 120],
    'max_depth': [6, 8, 10, 12, 15],
    'n_estimators': [100, 150, 200, 250],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    'min_child_samples': [20, 30, 40, 50],
    'min_split_gain': [0.0, 0.01, 0.05]
}

# 8. 网格搜索 + 十折交叉验证
Random_search = RandomizedSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
Random_search.fit(X_train, y_train)

# 9. 最佳模型
best_model = Random_search.best_estimator_
print("最佳参数:", Random_search.best_params_)

# 10. 测试集评估
y_test_pred = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
print(f"Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")

# 11. 预测 vs 实际 可视化
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual", linestyle="dashed")
plt.plot(y_test_pred, label="Predicted")
plt.legend()
plt.title("LightGBM Prediction vs Actual")
plt.savefig(os.path.join("D:\python\業務課題\picture", "lightgbm_prediction_vs_actual.svg"), format="svg")
plt.close()

# 12. 保存模型为 .pkl
joblib.dump(best_model, "lightgbm_best_model.pkl")
print("模型已保存为 lightgbm_best_model.pkl")
