import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Settings
lag=10           # How many lags do we use as features?
lstm_size=10     # Number of LSTM neurons
dense_size=10    # Number of fully-connected neurons
batch_size=128
epochs=500
patience=3       # Stop training if loss doesn't improve for this many epochs

# Import data
hsi = pd.read_csv("../Data/hsi.csv")
hsi["Date"] = pd.to_datetime(hsi["Date"])
hsi = hsi.dropna()

# Compute returns
data = (hsi["Adj Close"] /
         hsi.shift(1)["Adj Close"]
         - 1)
data = data.dropna()
data = data.to_numpy().reshape(-1,1)

# Standardize data
scalar = preprocessing.StandardScaler().fit(data)
data_std = scalar.transform(data).flatten()

input_data = data_std
target_data = data_std[lag:]

dataset = timeseries_dataset_from_array(
        input_data, target_data, 
        batch_size=batch_size,
        sequence_length=lag)

for batch in dataset.take(1):
    input_b, target_b = batch  
print("Input shape:", input_b.numpy().shape)
print("Target shape:", target_b.numpy().shape)    

# Set up layers 
inputs = Input(shape=(10,1))
x = LSTM(lstm_size, activation='tanh')(inputs)
x = Dense(dense_size, activation='tanh')(x)
predictions = Dense(1, activation='linear')(x)

# Callback for early stopping
callback = EarlyStopping(monitor='loss', patience=patience)

# Set up model
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='mean_squared_error',
              optimizer='adam')

print('Train...')
model.fit(dataset,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[callback])

# Evaluate
prediction_list = []
for batch in dataset:
    inputs, targets = batch  
    prediction_list.append(model.predict(inputs))
    
prediction = np.concatenate(prediction_list).flatten()
actual = target_data[-1*len(prediction):].flatten()
mse = np.mean(np.square(prediction - actual))
print("Prediction MSE:",mse)
