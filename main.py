import math
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas_datareader.data as pdr
import numpy as np
import pandas as pd
import yfinance as yf

yf.pdr_override()
coins = ['XRP-USD', 'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'DOT1-USD', 'BCH-USD', 'UNI3-USD', 'LTC-USD', 'XLM-USD']
# teže koliko bomo upoštevali kater indikator v optimizaciji format [SHARPE(3m), SHARPE(6m), SORTINO(3m)]
weights_ratios = [0.3, 0.3, 0.4]
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


def optimisation(coin_names, mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                 cov_matrix_6m, cov_matrix_poz_3m):
    weights_coins = np.zeros(len(coin_names))
    old_max = 0
    old_weights = weights_coins
    new_max = 0
    new_weights = weights_coins
    regression = 0.01
    change = (1 / (1 - regression)) * regression
    natancnost = 2
    for i in range(len(coin_names)):
        if i == 0:
            weights_coins[i] = 1
            # weights_coins /= np.sum(weights_coins)
            old_max = calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                                       cov_matrix_6m, cov_matrix_poz_3m, weights_coins)
            new_max = old_max
            new_weights = weights_coins.copy()
        else:
            old_weights = new_weights.copy()
            old_max = new_max
            weights_coins[weights_coins < 0.000001] = 0
            weights_coins /= np.sum(weights_coins)
            while True:
                weights_coins[i] += change
                weights_coins /= np.sum(weights_coins)
                new_max = calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                                           cov_matrix_6m, cov_matrix_poz_3m, weights_coins)
                new_weights = weights_coins.copy()
                if new_max < old_max:
                    weights_coins = old_weights.copy()
                    break
                else:
                    old_max = new_max
                    old_weights = new_weights.copy()

    for i in range(len(coin_names)-1, -1, -1):
        old_weights = new_weights.copy()
        old_max = new_max
        weights_coins[weights_coins < 0.000001] = 0
        weights_coins /= np.sum(weights_coins)
        while True:
            weights_coins[i] -= change
            weights_coins[weights_coins < 0.000001] = 0
            weights_coins /= np.sum(weights_coins)
            new_max = calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                                       cov_matrix_6m, cov_matrix_poz_3m, weights_coins)
            new_weights = weights_coins.copy()
            if new_max <= old_max:
                weights_coins = old_weights.copy()
                break
            else:
                old_max = new_max
                old_weights = new_weights.copy()

    print("Ratio: "+str(calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                                       cov_matrix_6m, cov_matrix_poz_3m, old_weights)))
    old_weights = np.round(old_weights, natancnost)
    old_weights /= np.sum(old_weights)
    return old_weights


def calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                     cov_matrix_6m, cov_matrix_poz_3m, weights_coins):
    weights_coins /= np.sum(weights_coins)
    returns_3m = np.sum(mean_return_3m * weights_coins) * 252
    returns_6m = np.sum(mean_return_6m * weights_coins) * 252
    # returns_poz_3m = np.sum(mean_return_poz_3m*weights_coins)*252
    std_dev_3m = np.sqrt(np.dot(weights_coins.T, np.dot(cov_matrix_3m, weights_coins))) * np.sqrt(252)
    std_dev_6m = np.sqrt(np.dot(weights_coins.T, np.dot(cov_matrix_6m, weights_coins))) * np.sqrt(252)
    std_dev_poz_3m = np.sqrt(np.dot(weights_coins.T, np.dot(cov_matrix_poz_3m, weights_coins))) * np.sqrt(252)
    risk_free_return = 0.012
    sharpe_3m = (returns_3m - risk_free_return) / std_dev_3m
    sharpe_6m = (returns_6m - risk_free_return) / std_dev_6m
    sortino_3m = (returns_3m - risk_free_return) / std_dev_poz_3m
    rez = sharpe_3m * weights_ratios[0] + sharpe_6m * weights_ratios[1] + sortino_3m * weights_ratios[2]
    return rez


prices_3m = get_data(coins, datum_3m, datum)
prices_6m = get_data(coins, datum_6m, datum)
day_returns_3m = prices_3m.pct_change()
day_returns_6m = prices_6m.pct_change()
day_returns_poz_3m = replace_negatives(prices_3m.pct_change())
mean_sharpe_3m = day_returns_3m.mean()
mean_sharpe_6m = day_returns_6m.mean()
mean_sortino_3m = day_returns_poz_3m.mean()
sharpe_3m_cov = day_returns_3m.cov()
sharpe_6m_cov = day_returns_6m.cov()
sortino_3m_cov = day_returns_poz_3m.cov()


results = optimisation(coins, mean_sharpe_3m, mean_sharpe_6m, mean_sortino_3m, sharpe_3m_cov, sharpe_6m_cov,
                       sortino_3m_cov)
print(pd.DataFrame(list(zip(coins, results*100)), columns=['Coins', 'Weights (%)']))
