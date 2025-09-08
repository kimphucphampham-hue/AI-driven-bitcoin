import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- STEP 1: DOWNLOAD AND MERGE DATA ---
print("Loading and merging data...")
try:
    # Download Bitcoin data
    df_price = pd.read_csv('bitcoin_ohlcv_with_volatility.csv')
    df_price.rename(columns={'timestamp': 'date'}, inplace=True)
    df_price['date'] = pd.to_datetime(df_price['date']).dt.date

    # Download sentiment data
    df_sentiment = pd.read_csv('daily_sentiment.csv')
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.date

    # Merge two DataFrames on date column
    df_combined = pd.merge(df_price, df_sentiment, on='date', how='inner')
    
    # Remove rows with empty values ​​(especially in volatility column)
    df_combined.dropna(inplace=True)
    
    print("Merge data successfully.")
    print("\nData table after merging:")
    print(df_combined.head())

except FileNotFoundError as e:
    print(f"Error: the file {e.filename} is not found. Please ensure CSV files are in the correct place.")
    exit()

# --- STEP 2: DATA PREPROCESSING FOR THE MODEL ---
# Prophet requires two columns: 'ds' (date) and 'y' (value to predict)
df_prophet = df_combined[['date', 'close', 'volatility_14d', 'avg_sentiment', 'volume']].copy()
df_prophet.rename(columns={'date': 'ds', 'close': 'y', 'volatility_14d': 'volatility', 'avg_sentiment': 'sentiment'}, inplace=True)

# Split the data into training and testing sets
# Use 80% for training and 20% for testing
train_size = int(len(df_prophet) * 0.8)
train_df = df_prophet[:train_size]
test_df = df_prophet[train_size:]

print(f"\nSize of training set: {len(train_df)} row")
print(f"Size of testing set: {len(test_df)} row")

# --- STEP 3: BUILD AND TRAIN THE MODEL ---
print("\nBuilding and training Prophet model...")

# Initialize the model
m = Prophet()

## Add external factors (sentiment and volatility)
m.add_regressor('volatility')
m.add_regressor('sentiment')
m.add_regressor('volume')

# Train model
m.fit(train_df)
print("Finish training model.")

# --- STEP 4: PREDICTION AND EVALUATION ---
print("\nPredicting on test set...")
# Create DataFrame to predict
future = m.make_future_dataframe(periods=len(test_df), include_history=False)

# Add volatility column and sentiment column from test_df into future DataFrame
future['volatility'] = test_df['volatility'].values
future['sentiment'] = test_df['sentiment'].values
future['volume'] = test_df['volume'].values

# Predict
forecast = m.predict(future)

# Evaluate model performance
y_true = test_df['y'].values
y_pred = forecast['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"\nEvaluate model performance on test set:")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")

print("\nThe process is complete!")