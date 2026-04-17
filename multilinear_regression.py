#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import fit_and_plot_linear, fit_and_plot_multi

DATA_FILE = Path(__file__).resolve().parent / "data" / "Advertising.csv"
df = pd.read_csv(DATA_FILE)
# print(df.head)

# Define an empty Pandas dataframe to store the R-squared value associated with each 
# predictor for both the train and test split
df_results = pd.DataFrame(columns=['Predictor', 'R2 Train', 'R2 Test'])


# For each predictor in the dataframe, call the function "fit_and_plot_linear()"
# from the helper file with the predictor as a parameter to the function.
# This function will split the data into train and test split, fit a linear model
# on the train data and compute the R-squared value on both the train and test data
for col in ['TV', 'Radio', 'Newspaper']:
    r2_train, r2_test = fit_and_plot_linear(df[[col]])
    df_results.loc[len(df_results)] = {
    'Predictor': col,
    'R2 Train': r2_train,
    'R2 Test': r2_test
}

# Call the function "fit_and_plot_multi()" from the helper to fit a multilinear model
# on the train data and compute the R-squared value on both the train and test data
r2_train, r2_test = fit_and_plot_linear(df[[col]])
df_results.loc[len(df_results)] = {
    'Predictor': 'Multi',
    'R2 Train': r2_train,
    'R2 Test': r2_test
}

# Store the R-squared values for all models
# in the dataframe intialized above


# Take a quick look at the dataframe
print(df_results)