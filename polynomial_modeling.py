#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Read the data from 'poly.csv' to a dataframe
df = pd.read_csv('data/poly.csv')

# print(df[['x']].values)
# print(df[['y']].vales)

# Get the column values for x & y in numpy arrays
x = df[['x']].values
y = df['y'].values

print(df.head())

# Plot x & y to visually inspect the data

fig, ax = plt.subplots()
ax.plot(x,y,'x')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.set_title('$y$ vs $x$');

# Fit a linear model on the data
model = LinearRegression()
model.fit(x,y)

# Get the predictions on the entire data using the .predict() function
y_lin_pred = model.predict(x)

plt.show()

# Now, we try polynomial regression
# GUESS the correct polynomial degree based on the above graph

guess_degree = 3

# Generate polynomial features on the entire data
x_poly= PolynomialFeatures(degree=guess_degree).fit_transform(x)