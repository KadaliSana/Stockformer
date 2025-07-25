import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from polygon import RESTClient
from torch.utils.data import DataLoader, TensorDataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Time2Vec(nn.Module):
    def __init__(self, output_dim):
        super(Time2Vec, self).__init__()
        self.linear = nn.Linear(1, output_dim)
        self.sin_layer = nn.Linear(1, output_dim)

    def forward(self, t):
        # Make sure t has the right shape [batch_size, seq_len, 1]
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        return self.linear(t) + torch.sin(self.sin_layer(t))

class Stockformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=20, num_layers=100):
        super(Stockformer, self).__init__()
        
        # Project input features to hidden_size
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.time_embedding = Time2Vec(hidden_size)
        
        # Using transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            batch_first=True,
            dropout=0.1  # Added dropout for regularization
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, time_input):
        # x shape: [batch_size, seq_len, input_size]
        # Project input to hidden dimension
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_size]
        
        ed = self.time_embedding(time_input)  # [batch_size, seq_len, hidden_size]
        
        # Add time embeddings to input
        x = x + ed  # [batch_size, seq_len, hidden_size]
        
        # Pass through transformer
        x = self.encoder(x)  # [batch_size, seq_len, hidden_size]
        
        # Take the last time step for prediction
        return self.fc(x[:, -1, :])  # [batch_size, output_size]

def save_model(model, optimizer, epoch, best_loss, model_path="stockformer.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved at {model_path}")

def load_model(model, optimizer, model_path="stockformer.pth"):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        model.eval()
        print(f"Model loaded from {model_path}")
        return epoch, best_loss
    except FileNotFoundError:
        print(f"No model found at {model_path}. Starting from scratch.")
        return 0, float("inf")


def read_stock_data_from_csv(file_path):
    """Reads stock data from a local CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at: {file_path}")

    df = pd.read_csv(file_path)
    
    # Ensure the timestamp column is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Ensure required columns are present
    required_columns = {"open", "high", "low", "close", "volume", "VWAP", "transactions", "OTC"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file missing required columns: {required_columns - set(df.columns)}")

    # Calculate Log Percent Change
    df["Kalman"]
    df["LogPercentChange"] = np.log(df["close"] / df["open"] + 1)

    return df[["LogPercentChange"]].dropna()


def create_rolling_dataset(df, window_size=32, prediction_steps=1):
    if len(df) <= window_size + prediction_steps:
        raise ValueError(f"DataFrame has {len(df)} rows, need at least {window_size + prediction_steps + 1}")

    X, y = [], []
    for i in range(len(df) - window_size - prediction_steps + 1):
        X.append(df.iloc[i:i + window_size].values)
        y.append(df.iloc[i + window_size + prediction_steps - 1].values[0])  # target is a scalar

    return np.array(X), np.array(y)


def prepare_dataset(csv_file_path, train_ratio=0.8, window_size=32):
    # Read stock data from CSV
    df = read_stock_data_from_csv(csv_file_path)
    if df.empty:
        raise ValueError(f"Failed to load data from {csv_file_path}")

    # Create dataset
    X, y = create_rolling_dataset(df, window_size=window_size)

    # Split train/test
    train_size = int(len(X) * train_ratio)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Convert to PyTorch tensors - reshape X to have feature dimension [batch, seq_len, features]
    X_train = torch.tensor(X_train, dtype=torch.float32).reshape(X_train.shape[0], X_train.shape[1], 1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # Make y a column vector
    X_test = torch.tensor(X_test, dtype=torch.float32).reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    # Create time inputs (0 to seq_len-1 for each sample)
    time_train = torch.arange(window_size, dtype=torch.float32).unsqueeze(0).repeat(X_train.shape[0], 1)
    time_test = torch.arange(window_size, dtype=torch.float32).unsqueeze(0).repeat(X_test.shape[0], 1)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train, time_train),
        batch_size=32,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test, time_test),
        batch_size=32,
        shuffle=False
    )

    print(f"Dataset prepared: {len(X_train)} training samples, {len(X_test)} testing samples")
    return train_loader, test_loader


def train_model(model, optimizer, criterion, train_loader, val_loader, 
                num_epochs=50, model_path="stockformer.pth"):
    # Try to load existing model
    start_epoch, best_loss = load_model(model, optimizer, model_path)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch, time_batch in train_loader:
            X_batch, y_batch, time_batch = X_batch.to(device), y_batch.to(device), time_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(X_batch, time_batch)  # X_batch already has correct shape [batch, seq, feature]
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch, time_batch in val_loader:
                X_batch, y_batch, time_batch = X_batch.to(device), y_batch.to(device), time_batch.to(device)
                predictions = model(X_batch, time_batch)  # X_batch already has correct shape
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        # Calculate average loss
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{start_epoch+num_epochs}, "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}")
    
    # Load the best model before returning
    load_model(model, optimizer, model_path)
    return best_loss

def backtest(model, test_loader):
    model.eval()
    total_profit = 1.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for X_batch, y_batch, time_batch in test_loader:
            X_batch, y_batch, time_batch = X_batch.to(device), y_batch.to(device), time_batch.to(device)
            
            # Get predictions
            predictions = model(X_batch, time_batch)  # X_batch already has correct shape
            
            # Calculate prediction accuracy (direction)
            # Extract the last value from X to compare with prediction
            last_values = X_batch[:, -1, 0].unsqueeze(-1)  # Get last time step's value and keep dims
            actual_direction = torch.sign(y_batch - last_values)
            predicted_direction = torch.sign(predictions - last_values)
            correct_predictions += (predicted_direction == actual_direction).sum().item()
            total_predictions += len(predictions)
            
            # Calculate profit
            price_change = y_batch - last_values
            profit_factor = 1 + (predicted_direction * price_change).cpu().numpy()
            
            # Apply simple trading fee (0.1%)
            profit_factor = np.where(predicted_direction.cpu().numpy() != 0, profit_factor - 0.001, profit_factor)
            
            total_profit *= np.mean(profit_factor)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return total_profit, accuracy

def main():
    # Set parameters
    ticker = "AAPL"
    hidden_size = 8
    num_heads = 2
    num_layers = 4
    learning_rate = 1e-4
    num_epochs = 2
    window_size = 32 

    try:
        train_loader, test_loader = prepare_dataset(ticker, window_size=window_size)
        
        # Initialize model
        model = Stockformer(
            input_size=1,
            hidden_size=hidden_size,
            output_size=1,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Train model
        print(f"Starting training for {ticker}...")
        best_loss = train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=num_epochs
        )

        save_model(model,optimizer,num_epochs,best_loss)

        # Backtest
        profit, accuracy = backtest(model, test_loader)
        print(f"Final backtest results for {ticker}:")
        print(f"Profit multiplier: {profit:.4f}x")
        print(f"Direction accuracy: {accuracy:.2%}")
        
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()