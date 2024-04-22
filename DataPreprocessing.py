# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 02:18:33 2024

@author: esola
"""

import requests
import pandas as pd
import json
import numpy as np
from datetime import datetime

symbols = ["AAPL", "CRVL", "EGHT", "IBM", "INTC", "ADBE", "MSFT", "NSIT",
           "PRO", "RMBS"]

data = []
for symbol in symbols:
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=' + symbol + '&outputsize=full&apikey=3FEF7XKMWV6LPTFJ'
    r = requests.get(url)
    data_Price = pd.DataFrame.from_dict(r.json()["Time Series (Daily)"],orient='index')
    
    data_ClosePrice = pd.DataFrame(data_Price["5. adjusted close"]).astype(float)
    data_ClosePrice = data_ClosePrice.rename(columns={"5. adjusted close": "Close"})
    
    data_Volume = pd.DataFrame(data_Price["6. volume"]).astype(float)
    data_Volume = data_Volume.rename(columns={"6. volume": "Volume"})
    
    
    f_DTE = open('DTE_' + symbol + '.json')
     
    load_DTE = json.loads(json.load(f_DTE))
    
    data_DTE = pd.DataFrame(columns = ["DebtToEquity"])
    
    for i in load_DTE:
        for j in i.keys():
            data_DTE.loc[j] = i[j]
    data_DTE = data_DTE.astype(float)
    
    f_EPS = open('EPS_' + symbol + '.json')
     
    load_EPS = json.load(f_EPS)["quarterlyEarnings"]
    
    data_EPS = pd.DataFrame(columns = ["EarningsPerShare"])
    
    for i in load_EPS:
            data_EPS.loc[i["reportedDate"]] = i["reportedEPS"]
    
    data_EPS = data_EPS.astype(float)
            
    url = 'https://www.alphavantage.co/query?function=EMA&symbol=' + symbol + '&interval=daily&outputsize=full&time_period=50&series_type=close&apikey=3FEF7XKMWV6LPTFJ'
    r = requests.get(url)
    data_EMA_50 = pd.DataFrame.from_dict(r.json()["Technical Analysis: EMA"],orient='index')
    data_EMA_50 = data_EMA_50.rename(columns={"EMA": "EMA 50 Period"})
    data_EMA_50 = data_EMA_50.astype(float)

    
    url = 'https://www.alphavantage.co/query?function=EMA&symbol=' + symbol + '&interval=daily&outputsize=full&time_period=200&series_type=close&apikey=3FEF7XKMWV6LPTFJ'
    r = requests.get(url)
    data_EMA_200 = pd.DataFrame.from_dict(r.json()["Technical Analysis: EMA"],orient='index')
    data_EMA_200 = data_EMA_200.rename(columns={"EMA": "EMA 200 Period"})
    data_EMA_200 = data_EMA_200.astype(float)

    
    url = 'https://www.alphavantage.co/query?function=RSI&symbol=' + symbol + '&interval=daily&outputsize=full&time_period=14&series_type=close&apikey=3FEF7XKMWV6LPTFJ'
    r = requests.get(url)
    data_RSI_14 = pd.DataFrame.from_dict(r.json()["Technical Analysis: RSI"],orient='index')
    data_RSI_14 = data_RSI_14.rename(columns={"RSI": "RSI 14 Period"})
    data_RSI_14 = data_RSI_14.astype(float)

    
    url = 'https://www.alphavantage.co/query?function=ADX&symbol=' + symbol + '&interval=daily&outputsize=full&time_period=14&series_type=close&apikey=3FEF7XKMWV6LPTFJ'
    r = requests.get(url)
    data_ADX_14 = pd.DataFrame.from_dict(r.json()["Technical Analysis: ADX"],orient='index')
    data_ADX_14 = data_ADX_14.rename(columns={"ADX": "ADX 14 Period"})
    data_ADX_14 = data_ADX_14.astype(float)

    
    url = 'https://www.alphavantage.co/query?function=OBV&symbol=' + symbol + '&interval=daily&outputsize=full&apikey=3FEF7XKMWV6LPTFJ'
    r = requests.get(url)
    data_OBV = pd.DataFrame.from_dict(r.json()["Technical Analysis: OBV"],orient='index')
    data_OBV = data_OBV.astype(float)

    url = 'https://www.alphavantage.co/query?function=AD&symbol=' + symbol + '&interval=daily&outputsize=full&apikey=3FEF7XKMWV6LPTFJ'
    r = requests.get(url)
    data_AD = pd.DataFrame.from_dict(r.json()["Technical Analysis: Chaikin A/D"],orient='index')
    data_AD = data_AD.astype(float)

    
    # Assuming data_EMA_50 and data_EMA_200 are your EMA DataFrames
    # Calculate the difference between the 50-day and 200-day EMA for each day
    ema_diff = data_EMA_50['EMA 50 Period'] - data_EMA_200['EMA 200 Period']

    # Determine the sign of the difference for each day (1 for positive, -1 for negative)
    ema_diff_sign = np.sign(ema_diff)

    # Calculate the difference in signs from one day to the next to find crossovers
    # A change in sign indicates a crossover
    ema_diff_sign_change = ema_diff_sign.diff()

    # Initialize a dictionary to store the crossover signals
    crossover_signals = {}

    # Iterate through the ema_diff_sign_change Series
    for date in ema_diff_sign.index[1:]:
        if ema_diff_sign_change[date] > 0:
            # If the sign change is positive, it indicates the 50-day EMA has crossed above the 200-day EMA
            crossover_signals[date] = 1  # Bullish crossover
        elif ema_diff_sign_change[date] < 0:
            # If the sign change is negative, it indicates the 50-day EMA has crossed below the 200-day EMA
            crossover_signals[date] = -1  # Bearish crossover
        else:
            # No crossover event
            crossover_signals[date] = 0

    # Convert the crossover signals dictionary into a DataFrame for easier handling
    crossover_signals_df = pd.DataFrame(list(crossover_signals.items()), columns=['Date', 'Crossover Signal'])
    crossover_signals_df.set_index('Date', inplace=True)
    
    DTE = {}
    date_DTE = np.flip(np.array([datetime.strptime(date,'%Y-%m-%d') for date in data_DTE.index]))
    for date in data_ClosePrice.index:
        if np.searchsorted(date_DTE, datetime.strptime(date, '%Y-%m-%d'))-1 >-1:
            DTE[date] = data_DTE.loc[date_DTE[np.searchsorted(date_DTE, datetime.strptime(date, '%Y-%m-%d'))-1].strftime('%Y-%m-%d')][0]
        else: 
            DTE[date] = np.NaN
        
    DTE = pd.DataFrame.from_dict(DTE,orient='index',columns = ["DebtToEquity"])
    DTE = DTE.astype(float)
    
    EPS = {}
    date_EPS = np.flip(np.array([datetime.strptime(date,'%Y-%m-%d') for date in data_EPS.index]))
    for date in data_ClosePrice.index:
        if np.searchsorted(date_EPS, datetime.strptime(date, '%Y-%m-%d'))-1 >-1:
            EPS[date] = data_ClosePrice.loc[date][0]/data_EPS.loc[date_EPS[np.searchsorted(date_EPS, datetime.strptime(date, '%Y-%m-%d'))-1].strftime('%Y-%m-%d')][0]
        else: 
            EPS[date] = np.NaN
    EPS = pd.DataFrame.from_dict(EPS,orient='index',columns = ["EarningsPerShare"])
    EPS = EPS.astype(float)
    
    
    data.append(pd.concat([data_ClosePrice, data_Volume, DTE, EPS, data_EMA_50, data_EMA_200, crossover_signals_df, data_RSI_14, data_ADX_14, data_OBV, data_AD], axis=1))
    
filename = "Risk_Analysis_Data_Adjusted.xlsx"
sheet_name = symbol 


with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
    for i,dt in enumerate(data):
        dt.to_excel(writer, sheet_name=symbols[i])












