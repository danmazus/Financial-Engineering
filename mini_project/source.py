import numpy as np
import pandas as pd

def dataframe_to_table(df, max_cols=10, **kwargs):
    num_cols = len(df.columns)

    if num_cols > max_cols:
        half_cols = max_cols // 2
        first_part = df.iloc[:, :half_cols]
        last_part = df.iloc[:, -half_cols:]

        # Create a DataFrame with one column of '...' that matches the number of rows
        ellipsis_df = pd.DataFrame(["..."] * first_part.shape[0], columns=["..."])

        # Concatenate first part, ellipsis, and last part
        df_subset = pd.concat([first_part, ellipsis_df, last_part], axis=1)

        print(
            f"DataFrame has {num_cols} columns, displaying the first {half_cols} and last {half_cols} columns with '...' in between.")
    else:
        df_subset = df

    latex_table = df_subset.to_latex(**kwargs)
    return latex_table

"""Cleaning and Setting Global Variables to use throughout"""
stocks = pd.read_csv('stock_prices.csv')
stocks = stocks.T
tickers = [heading.split()[0] for heading in stocks.index[1:]]
dates = stocks.iloc[0,1:].to_list()
print(stocks)

"""Cleaning and Setting Risk Free Rate"""
rates = pd.read_csv('daily_rates.csv')
rates = rates.T
rates = rates.iloc[:, ::-1]
rates.columns = range(len(rates.columns))
rates.iloc[1] = rates.iloc[1].astype(float)
print(rates)

# Conversion of rate to weekly
for j in range(len(rates.columns)):
    rates.iloc[1, j] = (1 + ((rates.iloc[1,j] / 100) / 365)) ** (365/52) - 1

print(rates)
print(dataframe_to_table(rates, max_cols=10))



# Convert Rates to a matrix
rates_matrix = rates.to_numpy()
rates_matrix = np.delete(rates_matrix, 0, 0)


# Transform into a matrix for computation purposes
price_matrix = stocks.to_numpy()
price_matrix = np.delete(price_matrix, 0, 0).astype(float)


"""COMPUTING EXCESS RETURN AND DE-MEANED RETURN MATRICES"""
# Initializing the excess return matrix as n x p
excess_return_matrix = np.zeros((9,26))

# Setting first column = 0 as there are no weekly returns for the first row
excess_return_matrix[:,0] = 0

# Setting amount of rows and columns for, for loop
rows, cols = price_matrix.shape

# Computing the weekly excess returns and storing it in the excess return matrix
for i in range(rows):
    for j in range(1, cols):
        excess_return_matrix[i,j] = (price_matrix[i,j] / price_matrix[i,j-1]) - 1 - rates_matrix[0,j]

# Deleting the first column as this has no excess returns (reference column)
excess_return_matrix = np.delete(excess_return_matrix, 0, 1)

# Converting the excess returns matrix to a pandas dataframe and setting title
excess_return_df = pd.DataFrame(excess_return_matrix, index=tickers)

# Initializing the de-meaned matrix and new rows and columns
Y = np.zeros((9,25))
rows_Y, cols_Y = Y.shape

# Computing the de-meaned matrix of weekly excess returns
for i in range(rows_Y):
    mean = np.mean(excess_return_matrix[i,:])
    for j in range(cols_Y):
        Y[i,j] = excess_return_matrix[i,j] - mean


"""COMPUTING HOLDINGS VECTOR C"""
# Compute sample Covariance Matrix
V = np.dot(Y, Y.T)/25

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

"""COMPUTING VARIANCE/STD DEV/EXCESS RETURN OF HOLDING VECTOR C"""
# Compute the variance using 1/e^TV^-1e as C is characteristic portfolio of e
weekly_var_c = np.dot(np.dot(h_C.T, V), h_C)
annual_var_c = np.dot(np.dot(h_C.T, V), h_C) * 52
weekly_std_c = np.sqrt(weekly_var_c)
annual_std_c = np.sqrt(weekly_var_c) * np.sqrt(52)
f_C_mean = np.mean(excess_return_matrix, axis=1)
f_C = np.dot(h_C.T, f_C_mean)
avg_f_C = np.mean(f_C)
avg_annual_f_C = avg_f_C * 52
annual_f_C = f_C * 52

"""COMPUTE VARIANCE OF INDIVIDUAL STOCKS"""
variances = np.diagonal(V)
annual_variances = variances * 52

"""PRINT STATEMENTS"""
# Printing the Excess Returns Dataframe
title = "Weekly Excess Returns for Given Stocks"
excess_return_df.title = title
print(f"\n{title}\n")
print(excess_return_df)


# Printing the De-Meaned Matrix in a Dataframe
de_mean_df = pd.DataFrame(Y, index=tickers)
title = "De-Meaned Matrix of Excess Returns"
de_mean_df.title = title
print(f"\n{title}\n")
print(de_mean_df)

# Printing the Covariance Matrix V
covariance_df = pd.DataFrame(V)
title="Covariance Matrix of Excess Returns"
covariance_df.title = title
print(f"\n{title}\n")
print(covariance_df)


# Convert into dataframe for viewing purposes
h_C_df = pd.DataFrame(h_C, index = tickers, columns = ['Holdings Percentage of each Stock'])
title = "Holdings Vector of Portfolio C with Given Stocks"
h_C_df.title = title
print(f"\n{title}\n")
print(h_C_df)
print(dataframe_to_table(h_C_df, max_cols = 6))

# Variances as dataframes
variances_df = pd.DataFrame(variances, index=tickers, columns=['Weekly Variance for Each Stock'])
annual_variances_df = pd.DataFrame(annual_variances, index=tickers, columns=['Annualized Variance for Each Stock'])

# Convert Expected excess returns into dataframe
f_C_df = pd.DataFrame(f_C, index = dates, columns = ['Expected Excess Returns Each Week'])
print(dataframe_to_table(f_C_df, max_cols=6, caption="Expected Excess Returns for Each Week in the Portfolio"))
annual_f_C_df = pd.DataFrame(annual_f_C, index = dates, columns = ['Annualized Expected Returns of Portfolio C for Each Week'])

print(f"\nWeekly Variance of Holding Vector C is: {weekly_var_c}")
print(f"\nAnnualized Variance of Holding Vector C is: {annual_var_c}")
print(f"\nWeekly Standard Deviation of Holding Vector C is: {weekly_std_c}")
print(f"\nAnnualized Standard Deviation of Holding Vector C is: {annual_std_c}")
print(f"\nAverage Weekly Expected Excess Return for Portfolio C is: {avg_f_C}")
print(f"\nAnnualized Average Weekly Expected Excess Return for Portfolio C is: {avg_annual_f_C}")
print(f"\n{f_C_df}")
print(f"\n{annual_f_C_df}")
print(f"\n{variances_df}")
print(f"\n{annual_variances_df}")
print(f_C_mean)
print(np.dot(h_C.T, f_C_mean.T))





