#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

def find_nearest(array,value):
    
    # print("\nInput array:  ", array, "Input value: ", value, "\n\n")
    # Hint: To find idx, use .idxmin() function on the series
    idx = pd.Series(np.abs(array - value)).idxmin()

    # Return the nearest neighbor index and value
    return idx, array[idx]

df_adv = pd.read_csv('data/Advertising.csv')
# print(df_adv.head())

# Get a subset of the data, e.g., rows 5 to 13
# Use the TV column as the predictor
x_true = df_adv.TV.iloc[5:13]
print(x_true)

# Use the Sales column as the response
y_true = df_adv.Sales.iloc[5:13]
print(y_true)

# Sort the data to get indices ordered from lowest to highest TV values
idx = np.argsort(x_true).values
# print(idx)

# Get the predictor data in the order given by idx above
x_true  = x_true.iloc[idx].values
print("x_true (TV budget)\n", x_true, "\n\n")

# Get the response data in the order given by idx above
y_true  = y_true.iloc[idx].values
print("y_true(Sales)\n", y_true, "\n\n")

# Create some synthetic x-values (might not be in the actual dataset)
# numpy.linspace() returns evenly spaced values over a specified interval
x = np.linspace(np.min(x_true), np.max(x_true))
# print("Output of linspace on x_true:", x, "\n\n")

# Initialize the y-values for the length of the synthetic x-values to zero
y = np.zeros((len(x)))

# Apply the KNN algorithm to predict the y-value for the given x value
for i, xi in enumerate(x):

    # Get the Sales values closest to the given x value
    y[i] = y_true[find_nearest(x_true, xi )[0]]
    
print("Output of y after KNN:", y, "\n\n")

# Plot the synthetic data along with the predictions    
plt.plot(x, y, '-.')

# Plot the original data using black x's.
plt.plot(x_true, y_true, 'kx')

# Set the title and axis labels
plt.title('TV vs Sales')
plt.xlabel('TV budget in $1000')
plt.ylabel('Sales in $1000')

plt.show()