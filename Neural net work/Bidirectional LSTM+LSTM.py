import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
def create_sliding_window_data(data, input_window, output_window, output_column):
    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i:i + input_window])
        y.append(data[i + input_window:i + input_window + output_window, output_column])
    return np.array(X), np.array(y)
 
 
# 读取 CSV 文件
data = pd.read_csv('./modified_processed_data1.csv')
 
# 数据预处理
data.drop('date', axis=1, inplace=True)
 
 
# 参数设置
input_window = 126
output_window = 1
output_column = 1
test_ratio = 0.2
 
# 划分训练集和测试集
data_len = len(data)
test_size = int(data_len * test_ratio)
 
X_train, X_test = data[:-test_size], data[-test_size:]
feature_data = X_train[['OT']]
feature_test_data = X_test[['OT']].reset_index(drop=True)
# 数据归一化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
feature_scaler = StandardScaler()
feature = feature_scaler.fit_transform(feature_test_data)
# feature_test = feature_scaler.transform(feature_test_data)
# 创建滑动窗口数据
X_train_sliding, y_train_sliding = create_sliding_window_data(X_train, input_window, output_window, output_column)
X_test_sliding, y_test_sliding = create_sliding_window_data(X_test, input_window, output_window, output_column)
 
# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_sliding, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_sliding, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_sliding, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_sliding, dtype=torch.float32)
 
# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
 
# 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional):
        super(LSTMModel, self).__init__()
        self.bidirectional_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.unidirectional_lstm = nn.LSTM(hidden_size * (2 if bidirectional else 1), hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        # Pass the input through the bidirectional LSTM
        out_bidirectional, _ = self.bidirectional_lstm(x)
 
        # Pass the output of the bidirectional LSTM through the unidirectional LSTM
        out_unidirectional, _ = self.unidirectional_lstm(out_bidirectional)
 
        # Apply ReLU activation on the output of the unidirectional LSTM
        out = F.relu(out_unidirectional[:, -1, :])
 
        # Pass the activated output through the fully connected layer
        out = self.fc(out)
        return out
 
 
# 初始化模型、优化器和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = X_train.shape[1]
hidden_size = 64
num_layers = 2
output_size = output_window
bidirectional = True
 
model = LSTMModel(input_size, hidden_size, num_layers, output_size, bidirectional).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()
losses = []
# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        losses.append(loss.item())
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
# 保存模型参数
torch.save(model.state_dict(), 'Bidirectional lstm_model.pth')
print("✅ 模型已保存为 Bidirectional lstm_model.pth")
results = []
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('lossPhoto')
 
# 滑窗预测
model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    predictions = model(X_test_tensor).squeeze().cpu().numpy()
 
# print(results)
predictions = feature_scaler.inverse_transform(predictions.reshape(1, -1)).tolist()[0]
# 注意 reshape 成适合 inverse_transform 的格式（和 predictions 保持一致）
#把 y_test_tensor 也 inverse_transform：
true = feature_scaler.inverse_transform(y_test_tensor.cpu().numpy().reshape(1, -1)).tolist()[0]
mse = mean_squared_error(true, predictions)
rmse = np.sqrt(mse)
print(f' RMSE: {rmse}')
mae = mean_absolute_error(true, predictions)
print(f'MAE: {mae}')
# 可视化预测结果与真实结果的对比
plt.figure(figsize=(10, 6))
plt.plot(true, label='True Values', color='blue')
plt.plot(predictions, label='Predictions', color='red')
plt.title('True vs Predicted')
plt.legend()
plt.show()
