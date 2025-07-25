import pandas as pd
import numpy as np
from polygon import RESTClient

def fetch_stock_data(ticker, start="2024-10-15", end="2024-12-31", save_path="stock_data2.csv"):
    client = RESTClient(api_key=POLYGON_API_KEY)
    all_data = []

    try:
        # Fetch data with pagination
        response = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="minute",
                from_=start,
                to=end,
                limit=50000,  # Max limit allowed per request
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(response)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker} in the specified date range")
        
        # Rename columns for clarity
        df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)

        # Convert timestamp to readable format
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Save as CSV file
        df.to_csv(save_path)
        print(f"✅ Data saved to {save_path}")
        
        return df
    
    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Example Usage
df = fetch_stock_data("AAPL")
print(df.head())  # Show first few rows
