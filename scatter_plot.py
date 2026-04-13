#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/Advertising.csv')

plt.scatter(df['TV'], df['Sales'])
plt.xlabel('TV Budget')
plt.ylabel('Sales')
plt.title('Sales of units by TV budget')
plt.show()