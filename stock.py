import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# =========================
# CONFIGURATION
# =========================
DATA_PATH = r"NSE-TATA.csv"
SEQ_LENGTH = 60
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_PATH = "lstm_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# DATA LOADING
# =========================
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

close_prices = df[["Close"]].values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

# =========================
# SEQUENCE CREATION
# =========================
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LENGTH)

train_size = int(len(X) * TRAIN_SPLIT)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# =========================
# DATASET CLASS
# =========================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODEL DEFINITION
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel().to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================
# TRAINING LOOP
# =========================
def train_model():
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.6f} "
              f"Val Loss: {avg_val_loss:.6f}")

train_model()

# =========================
# SAVE MODEL
# =========================
torch.save({
    "model_state_dict": model.state_dict(),
    "scaler": scaler
}, MODEL_PATH)

print("Model saved successfully.")

# =========================
# PREDICTION
# =========================
model.eval()
predictions = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(DEVICE)
        preds = model(X_batch)
        predictions.append(preds.cpu().numpy())

predictions = np.vstack(predictions)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test)

# =========================
# VISUALIZATION
# =========================
plt.figure(figsize=(14,6))
plt.plot(actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.legend()
plt.title("Stock Price Prediction")
plt.show()