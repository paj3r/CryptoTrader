import math
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas_datareader.data as pdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

yf.pdr_override()
coins = ['XRP-USD', 'LTC-USD', 'BCH-USD', 'XLM-USD', 'EOS-USD', 'XTZ-USD', 'ZRX-USD', 'OMG-USD']
# teže koliko bomo upoštevali kater indikator v optimizaciji format [SHARPE(3m), SHARPE(6m), SORTINO(3m)]
weights = [0.3, 0.3, 0.4]
datum = date.today()
datum_3m = datum + relativedelta(months=-3)
datum_6m = datum + relativedelta(months=-6)


def get_data(names, startdate, enddate):
    df = pdr.get_data_yahoo(names, startdate, enddate)['Adj Close']
    df.sort_index(inplace=True)
    return df


def replace_negatives(table):
    table[table < 0] = math.nan
    return table


prices_3m = get_data(coins, datum_3m, datum)
prices_6m = get_data(coins, datum_6m, datum)
day_returns_3m = prices_3m.pct_change()
day_returns_6m = prices_6m.pct_change()
day_returns_poz_3m = replace_negatives(prices_3m.pct_change())
mean_sharpe_3m = day_returns_3m.mean()
mean_sharpe_6m = day_returns_6m.mean()
mean_sortino_3m = day_returns_poz_3m.mean()
print(mean_sharpe_3m, mean_sortino_3m, mean_sharpe_6m)
