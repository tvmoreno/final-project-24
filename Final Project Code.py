import yfinance as yf
import pandas as pd
from datetime import datetime
import requests
import json
import csv
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from pathlib import Path


# S&P 500 / DJIA Collection & Calculation Section

stocks_dir = Path('Stocks')
stocks_dir.mkdir(parents=True, exist_ok=True)

def get_djia_symbols():
    """
    Gets the symbols of DJIA stocks from Wikipedia
    Returns list of symbols
    """
    dow_jones = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')[1]
    return dow_jones['Symbol'].tolist()


def get_sp500_symbols():
    """
    Gets the symbols of S&P 500 stocks from Wikipedia
    Returns list of symbols
    """
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500['Symbol'].tolist()


def get_market_caps(symbols):
    """
    Get the market capitalization for a list of stocks
    Returns dictionary with symbols as keys and market capitalization as values
    """
    market_caps = {}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        market_cap = stock.info.get('marketCap', None)
        if market_cap is not None:
            market_caps[symbol] = market_cap
        stock.history(period='5y').to_csv(f"Stocks/{symbol}_stock_history.csv")
    return market_caps


def filter_first_of_month(df):
    """
    Filters only the first month of data of market capitalization from csvs
    Returns a DataFrame with only the first trading day of each month
    """
    df = df.dropna().copy()
    df.set_index('Date', inplace=True)
    first_of_month = df.resample('MS').first()
    return first_of_month.reset_index()


def calculate_djia(filtered_data):
    """
    Calculates the DJIA values based on a list of stocks (added together)
    Returns a list of dates and DJIA overall value for those dates
    """
    print(filtered_data)
    dates = filtered_data[list(filtered_data.keys())[0]]['Date']
    djia_values = []

    for date in dates:
        total_close = 0
        for df in filtered_data.values():
            close_price = df[df['Date'] == date]['Close']
            if not close_price.empty:
                total_close += close_price.values[0]
        djia_values.append(total_close)

    initial_divisor = 0.14748071991788

    djia_values = [value / initial_divisor for value in djia_values]

    return dates, djia_values


def calculate_sp500(filtered_data, stocks_market_cap, sp500_symbols):
    """
    Calculates the S&P 500 values based on a list of stocks (added together)
    Returns a list of dates and S&P 500 overall value for those dates
    """
    dates = filtered_data[list(filtered_data.keys())[0]]['Date']
    sp500_values = []

    for date in dates:
        total_close = 0
        total_market_cap = 0
        for symbol, df in filtered_data.items():
            if symbol in sp500_symbols:
                close_price = df[df['Date'] == date]['Close']
                if not close_price.empty:
                    close_price_value = close_price.values[0]
                    market_cap = stocks_market_cap.get(symbol, 0)
                    total_close += close_price_value
                    total_market_cap += market_cap
        sp500_value = total_market_cap / total_close if total_close != 0 else 0
        sp500_values.append(sp500_value)

    return dates, sp500_values


cache_dir = Path.cwd() / "yfinance_cache"

dow_jones_symbols = get_djia_symbols()
sp500_symbols = get_sp500_symbols()
all_symbols = list(set(dow_jones_symbols + sp500_symbols))

start_year = 2019
end_year = 2024

start_date = datetime(2019, 5, 1)
end_date = datetime(2024, 5, 1)

stocks_market_cap = get_market_caps(all_symbols)

# Directory containing CSV files for all stocks
csv_directory = Path.cwd() / "Stocks"

# Load all stock data
all_data = {}
for file in csv_directory.iterdir():
    if file.suffix == '.csv' and file.stem.strip('_stock_history') in all_symbols:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        all_data[file.stem.strip('_stock_history')] = df

filtered_data = {symbol: filter_first_of_month(df) for symbol, df in all_data.items()}

# Ensure filtered_data has items before accessing keys
if filtered_data:
    djia_dates, djia_values = calculate_djia(filtered_data)
    sp500_dates, sp500_values = calculate_sp500(filtered_data, stocks_market_cap, sp500_symbols)

    djia_df = pd.DataFrame({'Date': djia_dates, 'DJIA': djia_values})
    sp500_df = pd.DataFrame({'Date': sp500_dates, 'S&P 500': sp500_values})

    djia_df.to_csv('djia_values.csv', index=False)
    sp500_df.to_csv('sp500_values.csv', index=False)

    djia_file_path = Path.cwd() / 'djia_values.csv'
    sp500_file_path = Path.cwd() / 'sp500_values.csv'

    print("DJIA and S&P 500 values saved to CSV files.")
else:
    print("No data available for DJIA and S&P 500.")

djia_dates, djia_values = calculate_djia(filtered_data)
sp500_dates, sp500_values = calculate_sp500(filtered_data, stocks_market_cap, sp500_symbols)

djia_df = pd.DataFrame({'Date': djia_dates, 'DJIA': djia_values})
sp500_df = pd.DataFrame({'Date': sp500_dates, 'S&P 500': sp500_values})

djia_df.to_csv('djia_values.csv', index=False)
sp500_df.to_csv('sp500_values.csv', index=False)

print("DJIA and S&P 500 values saved to CSV files.")

#

# CPI & Unemployment Calculation & Collection Section

# CPI
headers = {'Content-type': 'application/json'}

cpi_series_ids = ['CUUR0000SA0']

data = json.dumps({
    "seriesid": cpi_series_ids,
    "startyear": start_year,
    "endyear": end_year
})

response = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)

json_data = json.loads(response.text)

for series in json_data['Results']['series']:
    series_id = series['seriesID']
    rows = []

    for item in series['data']:
        year = item['year']
        period = item['period']
        value = item['value']
        footnotes = ",".join([footnote['text'] for footnote in item['footnotes'] if footnote])

        if 'M01' <= period <= 'M12':
            rows.append([series_id, year, period, value, footnotes])

    filename = f"CPI_{series_id}_{start_year}_{end_year}.csv"

    if rows:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["series id", "year", "period", "value", "footnotes"])
            writer.writerows(rows)
        print(f"CSV file {filename} has been created.")
    else:
        print(f"No data available for series {series_id}, skipping CSV creation.")

print("All CPI CSV files have been created.")


# Unemployment
start_year = "2019"
end_year = "2024"

headers = {'Content-type': 'application/json'}

unemployment_series_id = 'LNS14000000'

data = json.dumps({
    "seriesid": [unemployment_series_id],
    "startyear": start_year,
    "endyear": end_year
})

response = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)

json_data = json.loads(response.text)

for series in json_data['Results']['series']:
    series_id = series['seriesID']
    rows = []

    for item in series['data']:
        year = item['year']
        period = item['period']
        value = item['value']
        footnotes = ",".join([footnote['text'] for footnote in item['footnotes'] if footnote])

        if 'M01' <= period <= 'M12':
            rows.append([series_id, year, period, value, footnotes])

    filename = f"Unemployment_{series_id}_{start_year}_{end_year}.csv"

    if rows:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["series id", "year", "period", "value", "footnotes"])
            writer.writerows(rows)
        print(f"CSV file {filename} has been created.")
    else:
        print(f"No data available for series {series_id}, skipping CSV creation.")

cpi_file_path = Path.cwd() / f"CPI_CUUR0000SA0_{start_year}_{end_year}.csv"
unemployment_file_path = Path.cwd() / f"Unemployment_LNS14000000_{start_year}_{end_year}.csv"

print("All unemployment CSV files have been created.")

#

# DJIA, S&P 500, CPI, Unemployment Graphing


# Function to plot CPI data
def plot_cpi(file_path):
    """
    Takes in CPI data
    Produces a singular graph demonstrating data variation over time
    """
    cpi_data = pd.read_csv(file_path)

    cpi_data['year'] = cpi_data['year'].astype(str)
    cpi_data['period'] = cpi_data['period'].astype(str)
    cpi_data = cpi_data[cpi_data['period'].str.startswith('M')]
    cpi_data['Date'] = pd.to_datetime(cpi_data['year'] + cpi_data['period'].str[1:], format='%Y%m')

    cpi_data['CPI Change'] = cpi_data['value'].astype(float).diff()

    plt.figure(figsize=(10, 5))
    plt.plot(cpi_data['Date'], cpi_data['CPI Change'], label='Change in CPI')
    plt.xlabel('Date (YYYY-MM)')
    plt.ylabel('Change in CPI (%)')
    plt.title('Change in CPI Over Time (2019-2024)')
    plt.legend()
    plt.grid(True)
    months = MonthLocator(bymonth=[1, 4, 7, 10])
    months_fmt = DateFormatter('%Y-%m')
    ax = plt.gca()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    fig = plt.gcf()
    fig.autofmt_xdate()
    plt.show()


# Function to plot unemployment data
def plot_unemployment(file_path):
    """
    Takes in unemployment data
    Produces a singular graph demonstrating data variation over time
    """
    unemployment_data = pd.read_csv(file_path)

    unemployment_data['year'] = unemployment_data['year'].astype(str)
    unemployment_data['period'] = unemployment_data['period'].astype(str)
    unemployment_data = unemployment_data[unemployment_data['period'].str.startswith('M')]

    unemployment_data['Date'] = pd.to_datetime(unemployment_data['year'] + unemployment_data['period'].str[1:],
                                               format='%Y%m')
    plt.figure(figsize=(10, 5))
    plt.plot(unemployment_data['Date'], unemployment_data['value'].astype(float), label='Unemployment Rate')
    plt.xlabel('Date (YYYY-MM)')
    plt.ylabel('Unemployment Rate (%)')
    plt.title('Unemployment Rate Over Time (2019-2024)')
    plt.legend()
    plt.grid(True)
    months = MonthLocator(bymonth=[1, 4, 7, 10])
    months_fmt = DateFormatter('%Y-%m')
    ax = plt.gca()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    fig = plt.gcf()
    fig.autofmt_xdate()
    plt.show()


# Function to plot DJIA data
def plot_djia(file_path):
    """
    Takes in DJIA data
    Produces a singular graph demonstrating data variation over time
    """
    djia_data = pd.read_csv(file_path)
    djia_data['Date'] = pd.to_datetime(djia_data['Date'])
    plt.figure(figsize=(10, 5))
    plt.plot(djia_data['Date'], djia_data['DJIA'], label='DJIA')
    plt.xlabel('Date (YYYY-MM)')
    plt.ylabel('DJIA Value')
    plt.title('DJIA Over Time (2019-2024)')
    plt.legend()
    plt.grid(True)
    months = MonthLocator(bymonth=[1, 4, 7, 10])
    months_fmt = DateFormatter('%Y-%m')
    ax = plt.gca()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    fig = plt.gcf()
    fig.autofmt_xdate()
    plt.show()


# Function to plot S&P 500 data
def plot_sp500(file_path):
    """
    Takes in S&P 500 data
    Produces a singular graph demonstrating data variation over time
    """
    sp500_data = pd.read_csv(file_path)
    sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
    plt.figure(figsize=(10, 5))
    plt.plot(sp500_data['Date'], sp500_data['S&P 500'], label='S&P 500')
    plt.xlabel('Date (YYYY-MM)')
    plt.ylabel('S&P 500 Value')
    plt.title('S&P 500 Over Time (2019-2024)')
    plt.legend()
    plt.grid(True)
    months = MonthLocator(bymonth=[1, 4, 7, 10])
    months_fmt = DateFormatter('%Y-%m')
    ax = plt.gca()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    fig = plt.gcf()
    fig.autofmt_xdate()
    plt.show()


plot_cpi_file_path = Path.cwd() / "CPI_CUUR0000SA0_2019_2024.csv"
plot_unemployment_file_path = Path.cwd() / "Unemployment_LNS14000000_2019_2024.csv"
plot_djia_file_path = Path.cwd() / "djia_values.csv"
plot_sp500_file_path = Path.cwd() / "sp500_values.csv"

print("Plotting CPI data...")
plot_cpi(plot_cpi_file_path)

print("Plotting unemployment data...")
plot_unemployment(plot_unemployment_file_path)

print("Plotting DJIA data...")
plot_djia(plot_djia_file_path)

print("Plotting S&P 500 data...")
plot_sp500(plot_sp500_file_path)


# Comparative Graphs

# CPI & Unemployment
cpi_data = pd.read_csv(cpi_file_path)
unemployment_data = pd.read_csv(unemployment_file_path)

cpi_data['year'] = cpi_data['year'].astype(str)
cpi_data['period'] = cpi_data['period'].astype(str)
cpi_data = cpi_data[cpi_data['period'].str.startswith('M')]
cpi_data['Date'] = pd.to_datetime(cpi_data['year'] + cpi_data['period'].str[1:], format='%Y%m')
cpi_data = cpi_data[['Date', 'value']].rename(columns={'value': 'CPI'})

unemployment_data['year'] = unemployment_data['year'].astype(str)
unemployment_data['period'] = unemployment_data['period'].astype(str)
unemployment_data['Date'] = pd.to_datetime(unemployment_data['year'] + unemployment_data['period'].str[1:], format='%Y%m')
unemployment_data = unemployment_data[['Date', 'value']].rename(columns={'value': 'Unemployment'})

merged_data = pd.merge(cpi_data, unemployment_data, on='Date')

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Date (YYYY-MM)')
ax1.set_ylabel('Change in CPI', color='tab:blue')
ax1.plot(merged_data['Date'], merged_data['CPI'].diff(), color='tab:blue', label='Change in CPI')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Unemployment Rate (%)', color='tab:red')
ax2.plot(merged_data['Date'], merged_data['Unemployment'], color='tab:red', label='Unemployment Rate')
ax2.tick_params(axis='y', labelcolor='tab:red')

ax1.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
fig.autofmt_xdate()

fig.tight_layout()
plt.title('Change in CPI vs. Unemployment Rate Over Time (2019-2024)')
fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
plt.grid(True)
plt.show()


# DJIA & S&P 500
djia_data = pd.read_csv(djia_file_path)
sp500_data = pd.read_csv(sp500_file_path)

djia_data['Date'] = pd.to_datetime(djia_data['Date'])
djia_data = djia_data[['Date', 'DJIA']]

sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
sp500_data = sp500_data[['Date', 'S&P 500']]

merged_data = pd.merge(djia_data, sp500_data, on='Date')

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Date (YYYY-MM)')
ax1.set_ylabel('DJIA', color='tab:blue')
ax1.plot(merged_data['Date'], merged_data['DJIA'], color='tab:blue', label='DJIA')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('S&P 500', color='tab:red')
ax2.plot(merged_data['Date'], merged_data['S&P 500'], color='tab:red', label='S&P 500')
ax2.tick_params(axis='y', labelcolor='tab:red')

ax1.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
fig.autofmt_xdate()

fig.tight_layout()
plt.title('DJIA vs. S&P 500 Over Time (2019-2024)')
fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
plt.grid(True)
plt.show()


# S&P 500, Unemployment, and CPI
sp500_data['Date'] = sp500_data['Date'].dt.tz_localize(None)

merged_sp500_data = pd.merge(cpi_data, unemployment_data, on='Date')
merged_sp500_data = pd.merge(merged_sp500_data, sp500_data, on='Date')

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Date (YYYY-MM)')
ax1.set_ylabel('Change in CPI', color='tab:blue')
ax1.plot(merged_sp500_data['Date'], merged_sp500_data['CPI'].diff(), color='tab:blue', label='Change in CPI')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Unemployment Rate (%)', color='tab:red')
ax2.plot(merged_sp500_data['Date'], merged_sp500_data['Unemployment'], color='tab:red', label='Unemployment Rate')
ax2.tick_params(axis='y', labelcolor='tab:red')

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel('S&P 500', color='tab:green')
ax3.plot(merged_sp500_data['Date'], merged_sp500_data['S&P 500'], color='tab:green', label='S&P 500')
ax3.tick_params(axis='y', labelcolor='tab:green')

ax1.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
fig.autofmt_xdate()

fig.tight_layout()
plt.title('S&P 500, Unemployment Rate, and Change in CPI Over Time')
fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
plt.grid(True)
plt.show()


# DJIA, Unemployment, and CPI
djia_data['Date'] = djia_data['Date'].dt.tz_localize(None)

merged_djia_data = pd.merge(cpi_data, unemployment_data, on='Date')
merged_djia_data = pd.merge(merged_djia_data, djia_data, on='Date')

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Date (YYYY-MM)')
ax1.set_ylabel('Change in CPI', color='tab:blue')
ax1.plot(merged_djia_data['Date'], merged_djia_data['CPI'].diff(), color='tab:blue', label='Change in CPI')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Unemployment Rate (%)', color='tab:red')
ax2.plot(merged_djia_data['Date'], merged_djia_data['Unemployment'], color='tab:red', label='Unemployment Rate')
ax2.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Unemployment Rate (%)', color='tab:red')
ax2.plot(merged_djia_data['Date'], merged_djia_data['Unemployment'], color='tab:red', label='Unemployment Rate')
ax2.tick_params(axis='y', labelcolor='tab:red')

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel('DJIA', color='tab:green')
ax3.plot(merged_djia_data['Date'], merged_djia_data['DJIA'], color='tab:green', label='DJIA')
ax3.tick_params(axis='y', labelcolor='tab:green')

ax1.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
fig.autofmt_xdate()

fig.tight_layout()
plt.title('DJIA, Unemployment Rate, and CPI Over Time')
fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
plt.grid(True)
plt.show()

print("done.")
