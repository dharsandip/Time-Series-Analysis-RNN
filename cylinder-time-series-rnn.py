""""This data is from time-dependent solar load simulation done in ANSYS-Fluent CFD software
for a simple case of cylindrical room having top wall, bottom wall and cylindrical wall.
Solar flux is entering the room through the top wall which is semi-transparent and hitting other walls.
Other 2 walls are oqaque. Solar load depends on the time of the day, month, year, latitude and longitude
of a place etc. All these along with sun direction vector and solar irradiation are taken care of by Solar Calculator in ANSYS-Fluent. This time-dependent
problem is solved in ANSYS-Fluent using solar ray-tracing model with proper boundary conditions.
Simulation was done for 3600 secs (1 hour) and data (Area Weighted Average Static Temperature at bottom wall) was saved
for every sec. This data has been used here for time series Analysis of bottom wall temperature for last 9 minutes (540 secs) using 
RNN (Recurrent Neural Networks)"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training_dataset = pd.read_csv('Area-wt-avg-temp-outlet_trainingset.csv')
X_train = training_dataset.iloc[:, 1:2].values

test_dataset = pd.read_csv('Area-wt-avg-temp-outlet_testset.csv')
original_cfd_data = test_dataset.iloc[60:600, 0:2].values
X_time = original_cfd_data[:, 0]
y_temp = original_cfd_data[:, 1]


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler(feature_range = (0, 1))
sc_y = MinMaxScaler(feature_range = (0, 1))

X_train = sc_X.fit_transform(X_train)

# Creating a data structure with 60 timesteps and 1 output

X_train_new = []
y_train_new = []

for i in range(60, 3000):
    X_train_new.append(X_train[i-60:i])
    y_train_new.append(X_train[i])
X_train_new = np.array(X_train_new)
y_train_new = np.array(y_train_new)

X_train_new = np.reshape(X_train_new, (X_train_new.shape[0], X_train_new.shape[1], 1))

# Building the RNN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Adding the first LSTM layer
regressor.add(LSTM(units = 2, return_sequences = True, input_shape = (None, 1)))

# Adding the 2nd LSTM layer
regressor.add(LSTM(units = 2))
#regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

#compile the RNN

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train_new, y_train_new, epochs = 100, batch_size = 20)


# Making the predictions and visualising the results

dataset = pd.concat((training_dataset['Area-wt-avg Static Temperature at outlet (k)'], 
                    test_dataset['Area-wt-avg Static Temperature at outlet (k)']), axis = 0)


inputs = dataset[len(dataset) - len(test_dataset) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc_X.transform(inputs)


X_test_new = []

for i in range(60, 600):
    X_test_new.append(inputs[i-60:i])
X_test_new = np.array(X_test_new)
X_test_new = np.reshape(X_test_new, (X_test_new.shape[0], X_test_new.shape[1], 1))
predicted_temperature = regressor.predict(X_test_new)
predicted_temperature = sc_X.inverse_transform(predicted_temperature)


# Visualising the results

plt.plot(X_time, y_temp, color = 'red', label = 'Original Simulation Data')
plt.plot(X_time, predicted_temperature, color = 'blue', label = 'RNN Result')
plt.title('Time Series Analysis with RNN and comparison with original CFD simulation data')
plt.xlabel('Time')
plt.ylabel('Area-wt-avg Temperature at bottom wall')
plt.legend()
plt.show()


















