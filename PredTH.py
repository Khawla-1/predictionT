import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
# Load the CSV file
df = pd.read_csv('POWER.csv')

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['T2M', 'RH2M']])

# Define the number of time steps and features
n_steps = 12  # number of time steps to use in each sample
n_features = 2  # number of features (temperature and humidity)

# Split the dataset into input (X) and output (y) samples
X, y = [], []
for i in range(n_steps, len(df)):
    X.append(scaled_data[i-n_steps:i, :])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# Split the dataset into training and testing sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Define the Keras sequential model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
hist = model.fit(X_train, y_train, epochs=100, batch_size=32 , verbose=0, validation_split=0.5)

# Evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('test accuracy: %.3f' %acc)
# Make predictions
plt.title('learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.plot(hist.history['loss'], label= 'train')
plt.legend()
plt.show()