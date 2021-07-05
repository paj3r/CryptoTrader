import math
import pmdarima
import arch
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas_datareader.data as pdr
import pandas_ta as pta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn import linear_model


def get_closing(names, startdate, enddate):
    df = pdr.get_data_yahoo(names, startdate, enddate)['Adj Close']
    df.sort_index(inplace=True)
    return df


def get_volume(names, startdate, enddate):
    df = pdr.get_data_yahoo(names, startdate, enddate)['Volume']
    df.sort_index(inplace=True)
    return df


def get_marketCap(names, startdate, enddate):
    df = pdr.get_quote_yahoo(names, startdate, enddate)['marketCap']
    df.sort_index(inplace=True)
    return df


def get_4h_data(names, startdate, enddate):
    df = pdr.get_data_yahoo(names, startdate, enddate, interval='1h')['Adj Close']
    df.sort_index(inplace=True)
    df = df.resample('4H').last()
    return df


def replace_negatives(table):
    table[table < 0] = math.nan
    return table


def replace_zeroes(table):
    table[table < 0.10] = 0.0
    return table


def optimisation(coin_names, mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                 cov_matrix_6m, cov_matrix_poz_3m):
    weights_coins = np.zeros(len(coin_names))
    old_max = 0
    old_weights = weights_coins
    new_max = 0
    new_weights = weights_coins
    regression = 0.005
    change = (1 / (1 - regression)) * regression
    natancnost = 5
    natancnost_end = 1
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
            # weights_coins = np.round(weights_coins / np.linalg.norm(weights_coins, 1.0), natancnost)
            while True:
                weights_coins[i] += change
                weights_coins /= np.sum(weights_coins)
                # weights_coins = np.round(weights_coins / np.linalg.norm(weights_coins, 1.0), natancnost)
                new_max = calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                                           cov_matrix_6m, cov_matrix_poz_3m, weights_coins)
                new_weights = weights_coins.copy()
                if new_max < old_max:
                    weights_coins = old_weights.copy()
                    break
                else:
                    old_max = new_max
                    old_weights = new_weights.copy()

    for i in range(len(coin_names) - 1, -1, -1):
        old_weights = new_weights.copy()
        old_max = new_max
        weights_coins[weights_coins < 0.000001] = 0
        weights_coins /= np.sum(weights_coins)
        # weights_coins = np.round(weights_coins / np.linalg.norm(weights_coins, 1.0), natancnost)
        while True:
            weights_coins[i] -= change
            weights_coins[weights_coins < 0.000001] = 0
            weights_coins /= np.sum(weights_coins)
            # weights_coins = np.round(weights_coins / np.linalg.norm(weights_coins, 1.0), natancnost)
            new_max = calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                                       cov_matrix_6m, cov_matrix_poz_3m, weights_coins)
            new_weights = weights_coins.copy()
            if new_max <= old_max:
                weights_coins = old_weights.copy()
                break
            else:
                old_max = new_max
                old_weights = new_weights.copy()
    # print(old_weights)
    print("Ratio_pre: " + str(calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                                               cov_matrix_6m, cov_matrix_poz_3m, old_weights)))
    old_weights = np.round(old_weights / np.linalg.norm(old_weights, 1.0), 2)
    print("Ratio after rounding: " + str(
        calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                         cov_matrix_6m, cov_matrix_poz_3m, old_weights)))
    old_weights = replace_zeroes(old_weights)
    old_weights = np.round(old_weights / np.linalg.norm(old_weights, 1.0), 2)
    print("Ratio after >10% rule: " + str(
        calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                         cov_matrix_6m, cov_matrix_poz_3m, old_weights)))
    ratio =  calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                         cov_matrix_6m, cov_matrix_poz_3m, old_weights)
    if ratio < 2:
        return np.zeros(len(coin_names))
    return old_weights


def calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                     cov_matrix_6m, cov_matrix_poz_3m, weights_coins):
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


def calculate_RSI(prices, dolz):
    table = pd.DataFrame(list(zip(coins, np.zeros(len(coins)))), columns=['Coins', 'RSI'])
    table = np.zeros(len(coins))
    for index in range(len(coins)):
        rez = pta.rsi(prices[coins[index]], length=dolz)[-1]
        if rez > 70:
            table[index] = -1
        else:
            if rez < 30:
                table[index] = 1
            else:
                table[index] = 0
    return table


def arima_garch_prediction(prices, portfolio):
    predictions = np.zeros(len(coins))
    for i in range(len(coins)):
        if portfolio[i] == 0:
            continue
        returns = prices[coins[i]].pct_change().dropna()
        returns = returns * 100
        arima_model = pmdarima.auto_arima(returns)
        p, d, q = arima_model.order
        arima_residuals = arima_model.resid()

        garch = arch.arch_model(arima_residuals, p=1, q=1)
        garch_fitted = garch.fit(disp=0)

        predicted_mu = arima_model.predict(n_periods=1)[0]

        garch_forecast = garch_fitted.forecast(horizon=1, reindex=False)
        predicted_et = garch_forecast.mean['h.1'].iloc[-1]

        prediction = predicted_mu + predicted_et
        predictions[i] = prediction
    predictions[predictions > 0] = 1
    return predictions


def strategy():
    prices_3m = get_closing(coins, datum_3m, datum)
    prices_6m = get_closing(coins, datum_6m, datum)
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
    print(pd.DataFrame(list(zip(coins, results * 100)), columns=['Coins', 'Weights (%)']))
    return results


def linear_regression(prices, window_size, portfolio):
    predictions = np.zeros(len(coins))
    data = prices[-window_size:]
    for i in range(len(coins)):
        if portfolio[i] == 0:
            continue
        returns = data[coins[i]].pct_change().dropna()
        indices = np.array([*range(0, 59, 1)]).reshape(-1, 1)
        reg = linear_model.LinearRegression()
        reg.fit(indices, returns)
        predictions[i] = reg.coef_
    predictions[predictions > 0] = 1
    return predictions


def tactics(portfolio):
    prices_4h = get_4h_data(coins, datum_3m, datum)
    linear_predicts = linear_regression(prices_4h, 60, portfolio)
    print(pd.DataFrame(list(zip(coins, linear_predicts)), columns=['Coins', 'Linear Regression']))
    arga_predicts = arima_garch_prediction(prices_4h, portfolio)
    print(pd.DataFrame(list(zip(coins, arga_predicts)), columns=['Coins', 'ARIMA+GARCH']))
    rsi_predictions = calculate_RSI(prices_4h, 14)
    print(pd.DataFrame(list(zip(coins, rsi_predictions)), columns=['Coins', 'RSI']))
    actions = ["" for i in range(len(coins))]
    for i in range(len(coins)):
        if portfolio[i] == 0:
            actions[i] = "N/A"
            continue
        if (linear_predicts[i] == 1 or arga_predicts[i] == 1) and rsi_predictions[i] == 1:
            actions[i] = "BUY"
        if (linear_predicts[i] == 0 or arga_predicts[i] == 0) and rsi_predictions[i] == -1:
            actions[i] = "SELL"
        else:
            actions[i] = "WAIT"
    print(pd.DataFrame(list(zip(coins, actions)), columns=['Coins', 'Actions']))
    return actions


def tactics_test(prices, portfolio):
    linear_predicts = linear_regression(prices, 60, portfolio)
    print(pd.DataFrame(list(zip(coins, linear_predicts)), columns=['Coins', 'Linear Regression']))
    arga_predicts = arima_garch_prediction(prices, portfolio)
    print(pd.DataFrame(list(zip(coins, arga_predicts)), columns=['Coins', 'ARIMA+GARCH']))
    rsi_predictions = calculate_RSI(prices, 14)
    print(pd.DataFrame(list(zip(coins, rsi_predictions)), columns=['Coins', 'RSI']))
    actions = ["" for i in range(len(coins))]
    for i in range(len(coins)):
        if portfolio[i] == 0:
            actions[i] = "N/A"
            continue
        if (linear_predicts[i] == 1 or arga_predicts[i] == 1) and rsi_predictions[i] == 1:
            actions[i] = "BUY"
        if (linear_predicts[i] == 0 or arga_predicts[i] == 0) and rsi_predictions[i] == -1:
            actions[i] = "SELL"
        else:
            actions[i] = "WAIT"
    print(pd.DataFrame(list(zip(coins, actions)), columns=['Coins', 'Actions']))
    return actions


def strategy_test(prices_3m, prices_6m):
    # prices_3m = get_closing(coins, datum_3m, datum)
    # prices_6m = get_closing(coins, datum_6m, datum)
    tday_returns_3m = prices_3m.pct_change()
    tday_returns_6m = prices_6m.pct_change()
    tday_returns_poz_3m = replace_negatives(prices_3m.pct_change())
    tmean_sharpe_3m = tday_returns_3m.mean()
    tmean_sharpe_6m = tday_returns_6m.mean()
    tmean_sortino_3m = tday_returns_poz_3m.mean()
    tsharpe_3m_cov = tday_returns_3m.cov()
    tsharpe_6m_cov = tday_returns_6m.cov()
    tsortino_3m_cov = tday_returns_poz_3m.cov()
    results = optimisation(coins, tmean_sharpe_3m, tmean_sharpe_6m, tmean_sortino_3m, tsharpe_3m_cov, tsharpe_6m_cov,
                           tsortino_3m_cov)
    print(pd.DataFrame(list(zip(coins, results * 100)), columns=['Coins', 'Weights (%)']))
    return results


yf.pdr_override()
coins = ['XRP-USD', 'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'KMD-USD', 'BCH-USD', 'DOGE-USD', 'LTC-USD',
         'XLM-USD']
# teže koliko bomo upoštevali kater indikator v optimizaciji format [SHARPE(3m), SHARPE(6m), SORTINO(3m)]
weights_ratios = [0.5, 0.5, 0]
datum = date.today()
# print(datum)
start_date_test = datum + relativedelta(days=-729)
end_date_test = datum + relativedelta(days=-1)

test_1day_data = get_closing(coins, start_date_test+relativedelta(months=-6), end_date_test)
test_4h_data = get_4h_data(coins, start_date_test, end_date_test)
today = start_date_test + relativedelta(months=+3)
pct_profit = [[]]
profits = []
positions = np.zeros(len(coins))
buying_prices = np.zeros(len(coins))
test_port = np.zeros(len(coins))
today_plus_3m = today
while today < end_date_test:
    if today == today_plus_3m:
        print("Strategy change")
        test_port = strategy_test(test_1day_data[(today + relativedelta(months=-3)):today],
                              test_1day_data[(today + relativedelta(months=-6)):today])
        today_plus_3m = today + relativedelta(months=+3)
    if np.array_equal(test_port, np.zeros(len(coins))):
        today = today + relativedelta(months=+3)
        print("Bad strategy, skip 3 months")
        continue
    # print(test_4h_data[:today])
    prices_4h = test_4h_data[:today+relativedelta(days=+1)]
    print(today)
    for ix in range(-6, 0):
        cur_prices = prices_4h[:ix]
        actions = tactics_test(cur_prices, test_port)
        for i in range(0, len(coins)):
            cur_price = cur_prices.tail(1)[coins[i]].iloc[0]
            if test_port[i] == 0:
                continue
                # če se je cena znižala gremo vn
            if bool(cur_price < buying_prices[i]) and bool(positions[i] == 1):
                positions[i] = 0
                pricediff = buying_prices[i] - cur_price
                profits[i] += pricediff
                pct_profit[i].append((cur_price/buying_prices[i]))
                print("Panic sell " + coins[i])
                continue
            if bool(actions[i] == "BUY") and bool(positions[i] == 0):
                positions[i] = 1
                buying_prices[i] = cur_price
                print("Buy " + coins[i])
                continue
            if bool(actions[i] == "SELL") and bool(positions[i] == 1):
                positions[i] = 0
                pricediff = buying_prices[i] - cur_price
                profits[i] += pricediff
                pct_profit[i].append((cur_price / buying_prices[i]))
                print("Sell " + coins[i])
                continue
        today = today + relativedelta(days=+1)
        if today == today_plus_3m:
            cur_prices = prices_4h[:i]
            actions = tactics_test(cur_prices, test_port)
            for coin in range(0, len(coins)):
                cur_price = cur_prices.tail(1)[coins[i]]
                if test_port[i] == 0:
                    continue
                    # če se je cena znižala gremo vn
                if positions[i] == 1:
                    positions[i] = 0
                    pricediff = buying_prices[i] - cur_price
                    profits[i] += pricediff
                    pct_profit[i].append((cur_price / buying_prices[i]))
                    print("Selling all")
                    continue
    # actions = tactics_test(prices_4h, test_port)
    # print(actions)

f = open("profits.txt", "w")
f.write("".join(profits))
f.write("SUM:".join(sum(profits)))
f.close()
f = open("pctchange.txt", "w")
f.write("".join(pct_profit))
f.close()
datum_3m = datum + relativedelta(months=-3)
datum_6m = datum + relativedelta(months=-6)
# portfolio = strategy()
# actions = tactics()
