import pandas as pd

import numpy as np

def convert(file_path, output_path):
    """Reads a CSV file, converts timestamps to Unix time, calculates log returns and volatility,
    and applies a Kalman filter to smooth close and volatility."""
    df = pd.read_csv(file_path)
    
    # Convert timestamp to Unix time
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].astype(np.int64) // 10**9  

    # Calculate log returns and volatility
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    df["volatility"] = df["log_return"].rolling(window=30, min_periods=1).var().fillna(0)
    
    # Kalman filter parameters
    dt = 60  # time step
    # State vector: [close, velocity, volatility]
    S = np.array([[1, dt, 0],
                  [0, 1, 0],
                  [0, 0, 1]])  # volatility modeled as random walk
    
    # Observation matrix: we observe both close and volatility
    O = np.array([[1, 0, 0],
                  [0, 0, 1]])
    
    # Initial state estimate from first row
    I = np.array([df["close"].iloc[0], 0, df["volatility"].iloc[0]])
    Cov = np.diag([1, 1, 1])
    
    # Process and observation noise
    P_noise = np.diag([0.01, 0.01, 0.001])
    O_noise = np.diag([0.5, 0.1])
    
    # Initialize filtered columns
    df["kalman_close"] = np.nan
    df["kalman_vel"] = np.nan
    df["kalman_vol"] = np.nan
    df.loc[0, "kalman_close"] = I[0]
    df.loc[0, "kalman_vel"] = I[1]
    df.loc[0, "kalman_vol"] = I[2]
    
    for i in range(1, len(df)):
        # Prediction step
        I_0 = S @ I
        Cov_0 = S @ Cov @ S.T + P_noise
        
        # Build observation vector from current measurements
        z = np.array([df["close"].iloc[i], df["volatility"].iloc[i]])
        
        # Innovation and covariance
        y = z - (O @ I_0)
        S_cov = O @ Cov_0 @ O.T + O_noise
        
        # Kalman gain
        k = Cov_0 @ O.T @ np.linalg.inv(S_cov)
        
        # Update step
        I = I_0 + k @ y
        Cov = (np.eye(3) - k @ O) @ Cov_0
        
        # Store estimates
        df.loc[i, "kalman_close"] = I[0]
        df.loc[i, "kalman_vel"] = I[1]
        df.loc[i, "kalman_vol"] = I[2]
    
    df.to_csv(output_path, index=False)

convert("merged.csv", "unix.csv")
