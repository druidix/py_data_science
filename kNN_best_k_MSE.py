#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

def find_nearest(array,value):
    
    idx = pd.Series(np.abs(array - value)).idxmin()

    # Return the nearest neighbor index and value
    return idx, array[idx]

df_adv = pd.read_csv('data/Advertising.csv')

# Use the TV column as the predictor
x = df_adv[['TV']].values

# Use the Sales column as the response
y = df_adv[['Sales']].values


# Split the dataset in training and testing with 60% training set 
# and 40% testing set with random state = 66
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=66)

# Choose the minimum k value based on the instructions given on the left
k_value_min = 1

# Choose the maximum k value based on the instructions given on the left
k_value_max = 70

# Create a list of integer k values betwwen k_value_min and k_value_max using linspace
# Produces an array of ints from 1 through 70
k_list = np.linspace(k_value_min, k_value_max, 70, dtype=int)

# Set the grid to plot the values
fig, ax = plt.subplots(figsize=(10,6))

# Create a dictionary to store the k value against MSE fit {k: MSE@k} 
knn_dict = {}

# Variable used to alter the linewidth of each plot
j=0

# Loop over all the k values (array of 1 through 70)
for k_value in k_list:   
    
    # Creating a kNN Regression model 
    model = KNeighborsRegressor(n_neighbors=int(k_value))
    
    # Fitting the regression model on the training data 
    model.fit(x_train,y_train)
    
    # Use the trained model to predict on the test data 
    y_pred = model.predict(x_test)
    
    # Find the MSE
    MSE = mean_squared_error(y_test, y_pred)
    
    # Store the current MSE in the dict, keyed by the current k_value
    knn_dict[k_value] = MSE
    
    # Helper code to plot the data along with the model predictions
    colors = ['grey','r','b']
    if k_value in [1,10,70]:
        xvals = np.linspace(x.min(),x.max(),100).reshape(-1,1)
        ypreds = model.predict(xvals)
        ax.plot(xvals, ypreds,'-',label = f'k = {int(k_value)}',linewidth=j+2,color = colors[j])
        j+=1

ax.legend(loc='lower right',fontsize=20)
ax.plot(x_train, y_train,'x',label='train',color='k')
ax.plot(x_test, y_test,'x',label='train',color='k')
ax.set_xlabel('TV budget in $1000',fontsize=20)
ax.set_ylabel('Sales in $1000',fontsize=20)
plt.tight_layout()

# Find the lowest MSE among all the kNN models
min_mse = min(knn_dict.values())


# Use list comprehensions to find the k value associated with the lowest MSE
best_model = [key  for (key, value) in knn_dict.items() if value == min_mse]

# Print the best k-value
print ("The best k value is ",best_model,"with a MSE of ", min_mse)

# Plot a graph which depicts the relation between the k values and MSE
plt.figure(figsize=(8,6))

plt.plot(knn_dict.keys(), knn_dict.values(), 'k.-',alpha=0.5,linewidth=2)

# Set the title and axis labels
plt.xlabel('k',fontsize=20)
plt.ylabel('MSE',fontsize = 20)
plt.title('Test $MSE$ values for different k values - KNN regression',fontsize=20)
plt.tight_layout()
plt.show()