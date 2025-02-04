import numpy as np
import pandas as pd

stocks = pd.read_csv('stock_prices.csv')
print(stocks)

# Cleaning up the data to transform into a matrix for computation purposes
price_matrix = stocks.to_numpy()
price_matrix = np.delete(price_matrix, 0, 1).astype(float)
print(price_matrix)

# Initializing the excess return matrix as n x p
excess_return_matrix = np.zeros((26,9))

# Setting first row = 0 as there are no weekly returns for the first row
excess_return_matrix[0,:] = 0

# Setting amount of rows and columns for, for loop
rows, cols = price_matrix.shape

# Computing the weekly excess returns and storing it in the excess return matrix
for i in range(1, rows):
    for j in range(cols):
        excess_return_matrix[i,j] = (price_matrix[i,j] / price_matrix[i-1,j]) - 1

# Printing the excess return matrix
print("Excess Return Matrix")
print(excess_return_matrix)

# Converting the excess returns matrix to a pandas dataframe and setting title
excess_return_df = pd.DataFrame(excess_return_matrix,
                                columns = ['AAPL','MSFT','AMZN','NVDA','GOOGL','TSLA','META','BRK.B','UNH'])

title = "Weekly Excess Returns for Given Stocks"
excess_return_df.title = title

# Printing the dataframe
print(f"\n{title}\n")
print(excess_return_df)


# Initializing the de-meaned matrix
Y = np.zeros((26,9))

# Computing the de-meaned matrix of weekly excess returns
for j in range(cols):
    mean = np.mean(excess_return_matrix[1:,j])
    for i in range(1, rows):
        Y[i,j] = excess_return_matrix[i,j] - mean

Y = np.delete(Y, 0, 0)

V =  np.dot(Y, Y.T)/25

de_mean_df = pd.DataFrame(Y)
title = "De-Meaned Matrix of Excess Returns"
print(f"\n{title}\n")
print(de_mean_df)

print("\nCovariance Matrix V")
print(V)

# Compute inverse of V
V_inv = np.linalg.inv(V)

# Creating Vector e
e = np.ones(25)

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



