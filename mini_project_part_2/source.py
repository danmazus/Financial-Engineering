import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import os

def dataframe_to_table(df, max_cols=10, max_rows=10, **kwargs):
    num_cols = len(df.columns)
    num_rows = len(df)

    # Handling columns
    if num_cols > max_cols:
        half_cols = max_cols // 2
        first_part_cols = df.iloc[:, :half_cols]
        last_part_cols = df.iloc[:, -half_cols:]

        # Create a DataFrame with a single row of \cdots for columns
        ellipsis_col_df = pd.DataFrame({"$\cdots$": ["$\\cdots$"] * len(df)}, index=df.index)

        df = pd.concat([first_part_cols, ellipsis_col_df, last_part_cols], axis=1)

    # Handling rows
    if num_rows > max_rows:
        half_rows = max_rows // 2
        first_part_rows = df.iloc[:half_rows, :]
        last_part_rows = df.iloc[-half_rows:, :]

        # Create a DataFrame with one row of \vdots
        ellipsis_row_df = pd.DataFrame({col: ["$\\vdots$"] for col in df.columns})

        df = pd.concat([first_part_rows, ellipsis_row_df, last_part_rows], axis=0)

    latex_table = df.to_latex(escape=False, **kwargs)  # escape=False to allow LaTeX symbols
    return latex_table

file_path = \
    ('/Users/dannymazus/Documents/GitHub/Financial-Engineering/'
     'mini_project_part_2/data/sp500_market_caps.csv')
save_path = \
    ("/Users/dannymazus/Documents/GitHub/Financial-Engineering/"
     "mini_project_part_2/data/top_400_weekly_data.csv")



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

avg_risk_free_rate = 0.0427
avg_weekly_rf_rate = (1 + avg_risk_free_rate) ** (1 / 52) - 1

# Filling in our excess return matrix
for i in range(rows_close_price):
    for j in range(1, cols_close_price):
        excess_return_matrix[i,j] = (close_price_matrix[i,j] / close_price_matrix[i,j-1]) - 1 - avg_weekly_rf_rate

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
S_eigenvalues, S_eigenvectors = np.linalg.eigh(S)
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

# ================ Computing the James-Stein Estimator ========================
m_h = np.average(leading_eigenvector)

e = np.ones((p, 1))

s_2 = (1 / p) * np.sum(((np.sqrt(leading_eigenvalue) * leading_eigenvector) - (np.sqrt(leading_eigenvalue) * m_h)) ** 2)

nu_2 = l_2 * (1 / p)

c_jse = 1 - (nu_2 / s_2)

h_jse = m_h * e + c_jse * (leading_eigenvector - m_h * e)

V = (leading_eigenvector - l_2) * (np.dot(h_jse, h_jse.T) / np.dot(h_jse.T, h_jse)) + (n / p) * l_2 * np.eye(p)

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
annual_std_dev = weekly_std_dev * np.sqrt(52)

# Compute Expected excess returns f_C
exp_exc_returns_mean = np.mean(excess_return_matrix, axis=1)
exp_exc_returns = np.inner(h_C.T, exp_exc_returns_mean)
ann_exp_exc_returns = (1 + exp_exc_returns) ** 52 - 1

# Compute Variance of each stock
weekly_stock_variances = np.diag(S)
annual_stock_variances = weekly_stock_variances * 52


h_C_df = pd.DataFrame(h_C, index=close_prices_df.index, columns=['Percent Holdings'])

pd.set_option('display.max_rows', None)

# ================== Print Statements ========================

print(f'\nThe Holding Vector C is: \n{h_C_df}')
print(f"\nWeekly Variance of Holding Vector C is: {weekly_var_C}")
print(f"\nAnnualized Variance of Holding Vector C is: {annual_var_C}")
print(f"\nWeekly Standard Deviation of Holding Vector C is: {weekly_std_dev}")
print(f"\nAnnualized Standard Deviation of Holding Vector C is: {annual_std_dev}")
print(f"\nWeekly Expected Excess Return for Portfolio C is: {exp_exc_returns}")
print(f"\nAnnual Expected Excess Return for Portfolio C is: {ann_exp_exc_returns}")

# Getting the top n assets of our holdings
top_n_assets = 10
top_holdings_df = h_C_df.loc[h_C_df['Percent Holdings'].abs().sort_values(ascending=False).index].head(top_n_assets)

print(f'\nThe top {top_n_assets} Assets Holdings is: \n{top_holdings_df}')

# ============================== Plotting =======================================
# Barchart of Percentage Holdings
h_C_df_sort = h_C_df.sort_values(by=['Percent Holdings'], ascending=False)
h_C_df_sort = h_C_df_sort.reset_index()
plt.figure(figsize=(10, 8))
sns.barplot(x=h_C_df_sort['Symbol'], y=h_C_df_sort['Percent Holdings'], palette='viridis')
tick_positions = np.arange(0, len(h_C_df_sort['Percent Holdings']), 10)
plt.xticks(tick_positions, h_C_df_sort['Symbol'].iloc[tick_positions], rotation=90)
plt.xlabel('Symbol')
plt.ylabel('Percent Holdings')
plt.title('Portfolio Holdings Distribution')
plt.show()

# Histogram for distribution of percentage holdings
plt.figure(figsize=(10, 8))
sns.histplot(x=h_C_df_sort['Percent Holdings'], bins=50, kde=True, color='blue')
plt.xlabel('Percent Holdings')
plt.title('Histogram of Portfolio Holdings Distribution')
plt.show()

# Barchart of Stock Variances
plt.figure(figsize=(10, 8))
h_C_df = h_C_df.reset_index()
sns.barplot(x=h_C_df['Symbol'], y=annual_stock_variances, palette='viridis')
plt.plot(annual_var_C, color='black')
plt.xticks(tick_positions, h_C_df['Symbol'].iloc[tick_positions], rotation=90)
plt.xlabel('Symbol')
plt.ylabel('Annualized Stock Variance')
plt.title('Annualized Stock Variances for Each Stock')
plt.show()

# Barchart of Expected Excess Returns for Each Stock
plt.figure(figsize=(10, 8))
expected_mean_df = pd.DataFrame(exp_exc_returns_mean, index=close_prices_df.index, columns=['Expected Excess Returns'])
expected_mean_df_sort = expected_mean_df.sort_values(by=['Expected Excess Returns'], ascending=False)
expected_mean_df_sort = expected_mean_df_sort.reset_index()
sns.barplot(x=expected_mean_df_sort['Symbol'], y=expected_mean_df_sort['Expected Excess Returns'], palette='viridis')
plt.xticks(tick_positions, expected_mean_df_sort['Symbol'].iloc[tick_positions], rotation=90)
plt.xlabel('Symbol')
plt.ylabel('Expected Excess Returns')
plt.title('Expected Excess Returns for Each Stock')
plt.show()

# Distribution Histogram of Expected Excess Returns
plt.figure(figsize=(10, 8))
sns.histplot(x=expected_mean_df_sort['Expected Excess Returns'], bins=50, kde=True, color='red')
plt.xlabel('Expected Excess Returns')
plt.title('Distribution of Expected Excess Returns for Each Stock')
plt.show()

# Pie Chart of Long and Short Positions
long_positions = np.sum(h_C[h_C > 0])
short_positions = np.sum(np.abs(h_C[h_C < 0]))
size = [long_positions, short_positions]
plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(size, labels=['Long Positions', 'Short Positions'], autopct='%1.1f%%', explode=(0.01, 0.07), shadow=True, startangle=90, colors=('grey', 'orange'), wedgeprops={'edgecolor': 'k'}, textprops={'color': 'k'})
plt.title('Portfolio Allocation of Long and Short Positions')
plt.legend(wedges, ['Long', 'Short'],
           title='Positions', loc='best')
plt.setp(autotexts, size=12, weight='bold')
plt.show()


# Wealth over-time for our Portfolio
initial_wealth = 1000
portfolio_returns = np.dot(h_C.T, excess_return_matrix)
wealth_over_time = initial_wealth * (1 + portfolio_returns).cumprod()

# Wealth over-time for S&P 500
spx = yf.Ticker('^GSPC')
spx_data = spx.history(start='2024-03-31', end='2024-10-01')
spx_prices = spx_data['Close']
spx_prices.index = pd.to_datetime(spx_prices.index).date
spx_prices = spx_prices.reindex(close_prices_df.columns, method='ffill')
spx_returns = spx_prices.pct_change().dropna()
wealth_s_p = initial_wealth * (1 + spx_returns).cumprod()


# Wealth over-time for our portfolio vs the S&P 500
plt.figure(figsize=(10, 8))
plt.plot(excess_return_df.columns,wealth_over_time, label='Wealth Over Time', color='b')
plt.plot(wealth_s_p.index, wealth_s_p, label='S&P 500 Returns Given Initial Wealth', color='r')
plt.axhline(y=initial_wealth, color='k', linestyle='--', label=f'Initial Wealth: {initial_wealth}')
plt.xlabel('Date')
plt.ylabel('Wealth Over Time')
plt.legend()
plt.grid(True)
plt.show()