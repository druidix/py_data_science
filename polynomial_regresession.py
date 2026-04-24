#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from poly_helper import get_poly_pred
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Read the data from 'poly.csv' into a Pandas dataframe
df = pd.read_csv('data/poly.csv')

# Take a quick look at the dataframe
print(df.head())

# Get the column values for x & y as numpy arrays
x = df[['x']].values
y = df['y'].values

# Helper code to plot x & y to visually inspect the data
fig, ax = plt.subplots()
ax.plot(x,y,'x')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.set_title('$y$ vs $x$')
plt.show();

# Split the data into train and test sets
# Set the train size to 0.8 and random seed to 22
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state = 22 )

# Initialize a linear model
model = LinearRegression()

# Fit the model on the train data
model.fit(x_train, y_train)

# Get the predictions on the test data using the trained model
y_lin_pred = model.predict(x_test)

# Helper code to plot x & y to visually inspect the data
fig, ax = plt.subplots()
ax.plot(x,y,'x', label='data')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.plot(x_test, y_lin_pred, label='linear model predictions')
plt.legend()
plt.show()

### edTest(test_deg) ###
# Guess the correct polynomial degree based on the above graph
guess_degree = 3

# Predict on the entire polynomial transformed test data using helper function.
y_poly_pred = get_poly_pred(x_train, x_test, y_train, degree=guess_degree) 

# Helper code to visualise the results
idx = np.argsort(x_test[:,0])
x_test = x_test[idx]

# Use the above index to get the appropriate predicted values for y
# y values corresponding to sorted test data
y_test = y_test[idx]

# Linear predicted values  
y_lin_pred = y_lin_pred[idx]

# Non-linear predicted values
y_poly_pred= y_poly_pred[idx]

# First plot x & y values using plt.scatter
plt.scatter(x, y, s=10, label="Test Data")

# Plot the linear regression fit curve
plt.plot(x_test,y_lin_pred,label="Linear fit", color='k')

# Plot the polynomial regression fit curve
plt.plot(x_test, y_poly_pred, label="Polynomial fit",color='red', alpha=0.6)

# Assigning labels to the axes
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend()
plt.show()