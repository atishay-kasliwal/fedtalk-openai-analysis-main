import alpaca_trade_api as tradeapi

import datetime
import pandas as pd
import os

API_KEY = 'PKRABWZ5EFH71WFW8QHZ'
SECRET_KEY = 'd23n7ugBMSVlLyXhmELq9OUj5FXXC8falJyLs7bR'
BASE_URL = 'https://data.alpaca.markets'  # https://api.alpaca.markets
PRICE_DATA_BASE_PATH = "data_5Min/price/"
PRICE_BARS_INTERVAL = '1Min'
DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_price_change(start_time: datetime, end_time: datetime):
    start_date = datetime.datetime.strftime(start_time.date(), DATE_FORMAT)
    end_date = datetime.datetime.strftime(end_time.date(), DATE_FORMAT)
    
    price_data = get_price_data(start_date)
    if start_date != end_date:
        price_data = pd.concat([price_data, get_price_data(end_date)])  # Fix: use end_date instead of start_date again

    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], format=DATETIME_FORMAT)
    
    # Ensure we get the closest timestamps
    start_price = price_data.iloc[(price_data['timestamp'] - start_time).abs().argsort()[:1]]
    end_price = price_data.iloc[(price_data['timestamp'] - end_time).abs().argsort()[:1]]

    if start_price.empty or end_price.empty:
        return 0  # No valid data points found

    # Compute price change
    price_change = end_price['calculated_percentage'].values[0] - start_price['calculated_percentage'].values[0]

    return price_change

def get_price_volatility(start_time: datetime, end_time: datetime):
    start_date = datetime.datetime.strftime(start_time.date(), DATE_FORMAT)
    end_date = datetime.datetime.strftime(end_time.date(), DATE_FORMAT)
    price_data = get_price_data(start_date)
    if start_date != end_date:
        price_data = pd.concat([price_data, get_price_data(start_date)])
    price_data['calculated_volatility'] = price_data.apply(calculate_volatility, axis = 1)
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], format = DATETIME_FORMAT)
    price_data_filtered = price_data[price_data['timestamp'] == start_time]
    # print(f'{start_time} {end_time} {price_data_filtered}')
    price_data_max = price_data_filtered[price_data_filtered['calculated_volatility'].abs() == price_data_filtered['calculated_volatility'].abs().max()]
    return price_data_max['calculated_volatility'].values[0] if not price_data_max.empty else 0

def get_price_change_for_statements(start_time: datetime, end_time: datetime):
    start_date = datetime.datetime.strftime(start_time.date(), DATE_FORMAT)
    end_date = datetime.datetime.strftime(end_time.date(), DATE_FORMAT)
    price_data = get_price_data(start_date)
    if start_date != end_date:
        price_data = pd.concat([price_data, get_price_data(start_date)])
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], format = DATETIME_FORMAT)
    price_data_filtered = price_data[(price_data['timestamp'] >= start_time) & (price_data['timestamp'] < end_time)].sort_values(by = 'timestamp')
    if price_data_filtered.empty:
        return 0
    else:
        open_price = price_data_filtered['open'].values[0]
        close_price = price_data_filtered['close'].values[-1]
        return (close_price - open_price)/open_price

def get_price_data(date):
    date_file = PRICE_DATA_BASE_PATH + date + ".csv"
    if os.path.isfile(date_file):
        return pd.read_csv(date_file)
    
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

    start_date = date
    end_date = start_date

    # Retrieve data for 'SPY' stock
    historical_data = api.get_bars('SPY', PRICE_BARS_INTERVAL, start=start_date, end=end_date).df
    # Convert to UTC
    historical_data.index = historical_data.index.tz_convert(None)
    #print(historical_data)

    historical_data = historical_data.reset_index()

    historical_data['calculated_percentage'] = historical_data.apply(calculate_percentage_change, axis = 1)
    historical_data['calculated_volatility'] = historical_data.apply(calculate_volatility, axis = 1)

    #historical_data.to_csv(file_path, sep='\t')
    historical_data.to_csv(date_file)

    return historical_data


def calculate_percentage_change(row):
    return (row['close'] - row['open'])/row['open']

def calculate_volatility(row):
    return (row['high'] - row['low'])/row['low']