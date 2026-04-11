#!/usr/bin/env python3

import numpy as np
import pandas as pd

predictors = {
    "tv": [230.1, 44.5, 17.2, 151.5, 180.8],
    "radio": [37.8, 39.3, 45.9, 41.3, 10.8],
    "newspaper": [69.2, 45.1, 69.3, 58.5, 58.4]
}

outcomes = {
    "sales": [22.1, 10.4, 9.3, 18.5, 12.9]
}

X = pd.DataFrame(predictors)
Y = pd.DataFrame(outcomes)

print("Predictors DataFrame shape is: ", X.shape)
print("Outcomes DataFrame shape is: ", Y.shape)