import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset import ElectricityDataModule
from tpa_lstm import TPALSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

all_preds = []
all_targets = []
data_df = pd.read_csv(r"D:\python\業務課題\Dataset\ett.csv", index_col='date')
num_features = data_df.shape[1]

data_splits = {
    "train": 0.7,
    "val": 0.15,
    "predict": 0.15
}

pred_horizon = 4

elec_dm = ElectricityDataModule(
    dataset_splits=data_splits,
    batch_size=128,
    window_size=24,
    pred_horizon=pred_horizon,
    data_style="custom"
)

run_name = f"{pred_horizon}ts-kbest30"

hid_size = 64
n_layers = 1
num_filters = 3


name = f'{run_name}-TPA-LSTM'
checkpoint_loss_tpalstm = ModelCheckpoint(
    dirpath=f"checkpoints/{run_name}/TPA-LSTM",
    filename=name,
    save_top_k=1,
    monitor="val/loss",
    mode="min"
)

tpalstm_trainer = pl.Trainer(
    max_epochs=10,
    # accelerator='gpu',
    callbacks=[checkpoint_loss_tpalstm],
    strategy='auto',
    devices=1,
    # logger=wandb_logger_tpalstm
)

tpa_lstm = TPALSTM(
    input_size=num_features,
    hidden_size=hid_size,
    output_horizon=pred_horizon,
    num_filters=num_filters,
    obs_len=24,
    n_layers=n_layers,
    lr=1e-3
)

tpalstm_trainer.fit(tpa_lstm, elec_dm)

elec_dm.setup("predict")
run_to_load = run_name
model_path = f"checkpoints/{run_to_load}/TPA-LSTM/{name}.ckpt"
tpa_lstm = TPALSTM.load_from_checkpoint(model_path)
print(f"Model LSTM input size: {tpa_lstm.lstm.input_size}")

pred_dl = elec_dm.predict_dataloader()
y_pred = tpalstm_trainer.predict(tpa_lstm, pred_dl)

batch_idx = 0
start = 0
end = 5
for i, batch in enumerate(pred_dl):
    if start <= i <= end:
        inputs, labels = batch
        X, ytrue = inputs[batch_idx][:, -1], labels[batch_idx].squeeze()
        ypred = y_pred[i][batch_idx].squeeze()

        X = X.cpu().numpy()
        ytrue = ytrue.cpu().numpy()
        ypred = ypred.cpu().numpy()

        plt.figure(figsize=(8, 4))
        plt.plot(range(0, 24), X, label="Input")
        plt.scatter(range(24, 24 + pred_horizon), ytrue, color='cornflowerblue', label="True-Value")
        plt.scatter(range(24, 24 + pred_horizon), ypred, marker="x", color='green', label="TPA-LSTM pred")
        plt.legend(loc="lower left")

        # 儲存圖片到 picture 資料夾
        plt.savefig(f"D:/python/tpa-lstm-pytorch-main/picture/pred_{i}.png")
        plt.close()  # 關閉圖表，節省記憶體

                
        # 🔽 在這裡加入這兩行，收集數據
        all_preds.append(ypred)
        all_targets.append(ytrue)
    elif i > end:
        break

# 合并所有预测与真实值
all_preds = np.array(all_preds).reshape(-1)
all_targets = np.array(all_targets).reshape(-1)

# 计算指标
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
mae = mean_absolute_error(all_targets, all_preds)

print(f"✅ Evaluation on prediction batch {start}–{end}:")
print(f"  🔹 RMSE: {rmse:.4f}")
print(f"  🔹 MAE : {mae:.4f}")