import ccxt
import pandas as pd

# Initialize Binance exchange
binance = ccxt.binance()

# Get Bitcoin (BTC/USDT) OHLCV data for the last 50000 days
# '1d' is the time period in days (1 day)
print("Getting OHLCV data from Binance...")
try:
    ohlcv_data = binance.fetch_ohlcv('BTC/USDT', '1d', limit=50000)
    print("Get OHLCV successfully")
except Exception as e:
    print(f"Error when getting the data: {e}")
    exit()

# Change into DataFrame
df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Change timestamp into datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Set the 'timestamp' column as the index
df.set_index('timestamp', inplace=True)

# Calculate daily return from closing price
# pct_change() will calculate the percentage change from the previous line
df['daily_return'] = df['close'].pct_change()

# Calculate volatility using standard deviation of returns
# Use a 14-day rolling window
df['volatility_14d'] = df['daily_return'].rolling(window=14).std()

print("\nThe first 5 rows of the DataFrame have been updated:")
print(df.head())

# Save file CSV
df.to_csv('bitcoin_ohlcv_with_volatility.csv', index=True)
print("\nSaved data into 'bitcoin_ohlcv_with_volatility.csv' successfully!")