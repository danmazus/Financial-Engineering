import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import os

file_path = '/Users/dannymazus/Documents/GitHub/Financial-Engineering/mini_project_part_2/data/sp500_market_caps.csv'
save_path = "/Users/dannymazus/Documents/GitHub/Financial-Engineering/mini_project_part_2/data/top_400_weekly_data.csv"



# ================= Top 400 Stock Ticker Retrieval ======================= #

if os.path.exists(file_path) and os.path.exists(save_path):
    print("Both files exist, ...")

    print(f'\nLoading data from {file_path}')
    market_cap_df = pd.read_csv(file_path)
    print(f'\nData loaded from {file_path} successfully.')

    print(f'\nLoading data from {save_path}')
    weekly_data = pd.read_csv(save_path, index_col=[0, 1])
    print(f'\nData loaded from {save_path} successfully.')




else:
    print(f'Retreiving Data from Yahoo Finance and Wikipedia...')

    # Stock Retrieval
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    try:
        # Ensure SSL Verification is passed as True
        response = requests.get(url, verify=True)
        wiki_table = pd.read_html(response.text)
        sp_500_table = wiki_table[0]
    except Exception as e:
        print(f'Error retrieving data from S&P 500 data: {e}')

    sp_500_table['Symbol'] = sp_500_table['Symbol'].str.replace('.', '-', regex=False)
    tickers = sp_500_table['Symbol'].tolist()


    market_caps = {}

    for i, ticker in enumerate(tickers, start=1):
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap', None)
            if market_cap is not None:
                market_caps[ticker] = market_cap

            # Print progress update
            print(f"[{i}/{len(tickers)}] Processed: {ticker} - Market Cap: {market_cap}")

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    market_cap_df = pd.DataFrame(list(market_caps.items()), columns=['Symbol', 'Market Cap'])

    sp_500_table = sp_500_table.merge(market_cap_df, on='Symbol', how='left')

    os.makedirs('data', exist_ok=True)
    market_cap_df.to_csv(file_path, index=False)
    print(f'Data saved to {file_path}')

    # =================== Largest 400 Stock Selection ===============================
    top_400 = market_cap_df.sort_values(by='Market Cap', ascending=False).head(400)
    top_400_tickers = top_400['Symbol'].tolist()


    # ==================== Stock Data Retrieval ==========================

    # Approximately 26 weeks of data
    start_date = '2024-03-31'   # March 31, 2024
    end_date = '2024-10-01'     # October 1, 2024

    # Initialize Dictionary to Store Stock Information
    stock_info = {}

    print(f'Fetching data from {start_date} to {end_date}...')

    for i, ticker in enumerate(top_400_tickers, start=1):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(interval='1wk', start=start_date, end=end_date)

            if not data.empty:
                stock_info[ticker] = data

            print(f"[{i}/{len(top_400_tickers)}] Retrieved weekly data for {ticker}")

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    if stock_info:
        weekly_data = pd.concat(stock_info, names=['Symbol', 'Date'])
        weekly_data.to_csv(save_path)
        print(f'Data saved to {save_path}')
        print(f'Data Sample to ensure pulled correctly!')
        print(weekly_data.head())
    else:
        print(f'No stock data was retrieved. Check connection.')


# =================== Data Cleaning ==========================

# Resetting the index to access indexes as columns
weekly_data = weekly_data.reset_index()

# Fixing date-time to not include the time and just the date while making sure each date is in correct format
weekly_data['Date'] = pd.to_datetime(weekly_data['Date'], format='%Y-%m-%d').dt.date
# print("The cleaned weekly data is:")
# print(weekly_data)

# Pulling the close prices out of the dataframe and pivoting so that the index is by ticker and the column is the date
# with values of the close prices
close_prices_df = weekly_data.pivot_table(index='Symbol', columns='Date', values='Close')
# print('\nThe close prices of each stock for the given time period are:')
# print(close_prices_df)

# Converting the close price dataframe to a matrix to do computations
close_price_matrix = close_prices_df.to_numpy()
# print('\nThe converted close price matrix is:')
# print(close_price_matrix)

# =========================== Computations ===================================

# ================== Calculating Excess Return Matrix =========================

# Creating our excess return matrix of the same size as the closing price matrix
excess_return_matrix = np.zeros_like(close_price_matrix)

# Set the first column to 0 as this will be the reference column implying no excess returns
excess_return_matrix[:,0] = 0

# Getting the number of rows and columns to loop over to fill in the excess return matrix
rows_close_price, cols_close_price = close_price_matrix.shape

# Filling in our excess return matrix
for i in range(rows_close_price):
    for j in range(1, cols_close_price):
        excess_return_matrix[i,j] = (close_price_matrix[i,j] / close_price_matrix[i,j-1]) - 1

# Deleting the first column as this was the reference column
excess_return_matrix = np.delete(excess_return_matrix, 0, 1)

# Turning the excess return matrix into a dataframe with appropriate indices
excess_return_df = pd.DataFrame(excess_return_matrix, index=close_prices_df.index, columns=close_prices_df.columns[1:])

# Check whether there are NaN values
# rows_with_nan = np.where(np.isnan(excess_return_matrix).any(axis=1))[0]
# print(rows_with_nan)
# for row in rows_with_nan:
#     print(excess_return_df.iloc[row,:])

# Ensuring any of the NaN values are returned with 0 instead
excess_return_matrix = np.nan_to_num(excess_return_matrix, nan=0.0)


# ==================== Calculating Covariance Matrix ==========================

# Setting the number of observations and assets
rows_excess, cols_excess = excess_return_matrix.shape
n = cols_excess # Number of observations
p = rows_excess # Number of assets

# Calculating the mean and demeaned returns
mean_returns = np.mean(excess_return_matrix, axis=1, keepdims=True)
demeaned_returns = excess_return_matrix - mean_returns

# Calculating the Covariance Matrix
S = np.dot(demeaned_returns, demeaned_returns.T) / n


# ================= Computing the Single Factor Covariance Matrix, Sigma =========================

# Computing the Eigenvalues and Eigenvectors
S_eigenvalues, S_eigenvectors = np.linalg.eig(S)
S_eigenvalues, S_eigenvectors = S_eigenvalues.real, S_eigenvectors.real

# Getting the leading eigenvalue and corresponding eigenvector
leading_index = np.argmax(np.abs(S_eigenvalues))
leading_eigenvalue = S_eigenvalues[leading_index]
leading_eigenvector = S_eigenvectors[:, leading_index].reshape(-1, 1)

# Computing the trace and l_2
trace_S = np.trace(S)
l_2 = (trace_S - leading_eigenvalue) / (n - 1)

# Computing sigma
sigma = (leading_eigenvalue - l_2) * np.dot(leading_eigenvector, leading_eigenvector.T) + (n/p) * l_2 * np.eye(p)

# ================== Holdings Vector Calculation ====================

# Compute the Inverse of sigma
sigma_inv = np.linalg.inv(sigma)

# Setup the e vector
e = np.ones(p)

# Compute the numerator of h_C
numer = np.dot(sigma_inv, e)

# Compute the denominator of h_C
denom = np.dot(np.dot(e.T, sigma_inv), e)

# Compute the holdings vector h_C
h_C = numer / denom


# ================= Compute the needed statistics of the portfolio ===================

# Compute Weekly Variance
weekly_var_C = np.dot(np.dot(h_C.T, sigma), h_C)

# Compute annual variance
annual_var_C = weekly_var_C * 52

# Compute Weekly Standard Deviation
weekly_std_dev = np.sqrt(weekly_var_C)

# Compute Annual Standard Deviation
annual_std_dev = weekly_std_dev * 52

# Compute Expected excess returns f_C
exp_exc_returns_mean = np.mean(excess_return_matrix, axis=1)
exp_exc_returns = np.dot(h_C.T, exp_exc_returns_mean)

# Compute Variance of each stock
weekly_stock_variances = np.diag(S)
annual_stock_variances = weekly_stock_variances * 52


h_C_df = pd.DataFrame(h_C, index=close_prices_df.index, columns=['Percent Holdings'])

pd.set_option('display.max_rows', None)

print(f'\nThe Holding Vector C is: \n{h_C_df}')
print(f"\nWeekly Variance of Holding Vector C is: {weekly_var_C}")
print(f"\nAnnualized Variance of Holding Vector C is: {annual_var_C}")
print(f"\nWeekly Standard Deviation of Holding Vector C is: {weekly_std_dev}")
print(f"\nAnnualized Standard Deviation of Holding Vector C is: {annual_std_dev}")
print(f"\nWeekly Expected Excess Return for Portfolio C is: {exp_exc_returns}")

top_n_assets = 10

top_holdings_df = h_C_df.sort_values(by='Percent Holdings', ascending=False).head(top_n_assets)

plt.figure(figsize = (8, 8))
plt.pie(top_holdings_df['Percent Holdings'], labels=top_holdings_df.index, autopct='%1.1f%%', startangle=90)
plt.title(f'Top {top_n_assets} Asset Holdings')
plt.show()






