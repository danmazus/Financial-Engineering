import numpy as np
import pandas as pd

stocks = pd.read_csv('/Users/dannymazus/Documents/GitHub/Financial-Engineering/mini_project/stock_prices.csv')
stocks = stocks.T
print(stocks)

# Cleaning up the data to transform into a matrix for computation purposes
price_matrix = stocks.to_numpy()
price_matrix = np.delete(price_matrix, 0, 0).astype(float)
print(price_matrix)

# Initializing the excess return matrix as n x p
excess_return_matrix = np.zeros((9,26))

# Setting first column = 0 as there are no weekly returns for the first row
excess_return_matrix[:,0] = 0

# Setting amount of rows and columns for, for loop
rows, cols = price_matrix.shape

# Computing the weekly excess returns and storing it in the excess return matrix
for i in range(rows):
    for j in range(1, cols):
        excess_return_matrix[i,j] = (price_matrix[i,j] / price_matrix[i,j-1]) - 1

excess_return_matrix = np.delete(excess_return_matrix, 0, 1)

# Converting the excess returns matrix to a pandas dataframe and setting title
excess_return_df = pd.DataFrame(excess_return_matrix)

# Printing the Excess Returns Dataframe
title = "Weekly Excess Returns for Given Stocks"
excess_return_df.title = title
print(f"\n{title}\n")
print(excess_return_df)

# Initializing the de-meaned matrix
Y = np.zeros((9,25))
rows_Y, cols_Y = Y.shape

# Computing the de-meaned matrix of weekly excess returns
for i in range(rows_Y):
    mean = np.mean(excess_return_matrix[i,:])
    for j in range(cols_Y):
        Y[i,j] = excess_return_matrix[i,j] - mean

# Compute sample Covariance Matrix
V = np.dot(Y, Y.T)/25

# Printing the De-Meaned Matrix in a Dataframe
de_mean_df = pd.DataFrame(Y)
title = "De-Meaned Matrix of Excess Returns"
print(f"\n{title}\n")
print(de_mean_df)

# Printing the Covariance Matrix V
print("\nCovariance Matrix V")
print(V)

# Compute inverse of V
V_inv = np.linalg.inv(V)

# Creating Vector e
e = np.ones(9)

# Compute numerator of h_C
numer = np.dot(V_inv, e)

# Compute denominator of h_C
denom = np.dot(np.dot(e.T, V_inv), e)

# Calculate the holdings vector, h_C
h_C = numer/denom

# Convert into dataframe for viewing purposes
h_C_df = pd.DataFrame(h_C, columns = ['Holdings Percentage of each Stock'])
title = "Holdings Vector of Portfolio C with Given Stocks"
print(f"\n{title}\n")
print(h_C_df)


