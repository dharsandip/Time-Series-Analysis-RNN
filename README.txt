This data is from time-dependent solar load simulation done in ANSYS-Fluent CFD software for a simple case of cylindrical room having top wall, bottom wall and cylindrical wall.
Solar flux is entering the room through the top wall which is semi-transparent and hitting other walls.
Other 2 walls are oqaque. Solar load depends on the time of the day, month, year, latitude and longitude
of a place etc. All these are taken care of by Solar Calculator in ANSYS-Fluent. This time-dependent
problem is solved in ANSYS-Fluent using solar ray-tracing model with proper boundary conditions.
Simulation was done for 3600 secs (1 hour) and data (Area Weighted Average Static Temperature at outlet) was saved
for every sec. This data has been used here for time series Analysis of outlet temperature for last 9 minutes (540 secs) using RNN (Recurrent Neural Networks).



mean_squared_error of training dataset (loss): 5.0769e-05

