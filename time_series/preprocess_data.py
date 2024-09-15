import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Read the datasets
coinbase = pd.read_csv('coinbase.csv')
bitstamp = pd.read_csv('bitstamp.csv')

# Convert 'Timestamp' to datetime
coinbase['timestamp'] = pd.to_datetime(coinbase['Timestamp'], unit='s')
bitstamp['timestamp'] = pd.to_datetime(bitstamp['Timestamp'], unit='s')

# Drop the original 'Timestamp' columns
coinbase.drop(columns=['Timestamp'], inplace=True)
bitstamp.drop(columns=['Timestamp'], inplace=True)

# Merge datasets on 'timestamp' with an outer join
data = pd.merge(
    coinbase,
    bitstamp,
    on='timestamp',
    suffixes=('_coinbase', '_bitstamp'),
    how='outer'
)

# Sort by timestamp
data.sort_values('timestamp', inplace=True)

# Set 'timestamp' as index
data.set_index('timestamp', inplace=True)

# Features to average
features = [
    'Open', 'High', 'Low', 'Close',
    'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price'
]

# Average the features from both exchanges
for feature in features:
    data[feature] = data[[f"{feature}_coinbase",
                          f"{feature}_bitstamp"]].mean(axis=1)

# Drop individual exchange columns
data.drop(
    columns=[col for col in data.columns if '_coinbase' in col or '_bitstamp' in col],
    inplace=True
)

# Interpolate missing values
data.interpolate(method='time', inplace=True)

# Drop remaining NaN values
data.dropna(inplace=True)

# Select relevant features
data = data[['Close', 'Volume_(BTC)']]

# Rescale the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Save the preprocessed data
np.save('preprocessed_data.npy', data_scaled)
