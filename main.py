import math
import pmdarima
import arch
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas_datareader.data as pdr
import pandas_ta as pta
import numpy as np
import pandas as pd
import yfinance as yf
import pytz


def get_1day_data(names, startdate, enddate):
    df = pdr.get_data_yahoo(names, startdate, enddate)['Adj Close']
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
    for i in range(len(coin_names)):
        if i == 0:
            weights_coins[i] = 1
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

    for i in range(len(coin_names) - 1, -1, -1):
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
    print("Ratio_pre: " + str(calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                                               cov_matrix_6m, cov_matrix_poz_3m, old_weights)))
    old_weights = np.round(old_weights / np.linalg.norm(old_weights, 1), 2)
    print("Ratio after rounding: " + str(
        calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                         cov_matrix_6m, cov_matrix_poz_3m, old_weights)))
    old_weights = replace_zeroes(old_weights)
    old_weights = np.round(old_weights / np.linalg.norm(old_weights, 1), 2)
    print("Ratio after >10% rule: " + str(
        calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                         cov_matrix_6m, cov_matrix_poz_3m, old_weights)))
    rat = calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                           cov_matrix_6m, cov_matrix_poz_3m, old_weights)
    if rat < 1:
        return np.zeros(len(coin_names))
    return old_weights


def calculate_ratios(mean_return_3m, mean_return_6m, mean_return_poz_3m, cov_matrix_3m,
                     cov_matrix_6m, cov_matrix_poz_3m, weights_coins):
    if np.array_equal(weights_coins, np.zeros(len(coins))):
        return 0
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
    table = np.zeros(len(coins))
    for index in range(len(coins)):
        rez = pta.rsi(prices[coins[index]], length=dolz)[-1]
        rez2 = pta.rsi(prices[coins[index]], length=dolz)[-2]
        if rez < 70 and rez2 > rez:
            table[index] = 1
        else:
            table[index] = 0
    return table


def calculate_RSI_bear(prices, dolz):
    table = np.zeros(len(coins))
    for index in range(len(coins)):
        rez = pta.rsi(prices[coins[index]], length=dolz)[-1]
        rez2 = pta.rsi(prices[coins[index]], length=dolz)[-2]
        if rez < 30 and rez2 < rez:
            table[index] = 1
        else:
            table[index] = 0
    return table


def calculate_AMA(prices, short, far):
    table = np.zeros(len(coins))
    for index in range(len(coins)):
        loc_pric = prices[coins[index]]
        a = np.std(loc_pric[-short:])
        b = np.std(loc_pric[-far:])
        v = b / a + short
        p = int(round(v))
        cut_pric = loc_pric[-p:]
        k = np.sum(cut_pric)
        ama = k / v
        if ama < loc_pric[-1]:
            table[index] = 1
        else:
            table[index] = 0
    return table


def arima_garch_prediction(prices, portfo, window_size):
    predictions = np.zeros(len(coins))
    for i in range(len(coins)):
        if portfo[i] == 0:
            continue
        returns = prices[coins[i]].pct_change().dropna()
        returns = returns[-window_size:]
        returns = returns * 100
        arima_model = pmdarima.auto_arima(returns)
        p, d, q = arima_model.order
        arima_residuals = arima_model.resid()

        garch = arch.arch_model(arima_residuals, p=1, q=1)
        garch_fitted = garch.fit(disp='off')

        predicted_mu = arima_model.predict(n_periods=1)[0]

        garch_forecast = garch_fitted.forecast(horizon=1, reindex=False)
        predicted_et = garch_forecast.mean['h.1'].iloc[-1]

        prediction = predicted_mu + predicted_et
        predictions[i] = prediction
    predictions[predictions > 0] = 1
    predictions[predictions < 0] = -1
    return predictions


def tactics(prices, portfo):
    arga_predicts = arima_garch_prediction(prices, portfo, 120)
    print(pd.DataFrame(list(zip(coins, arga_predicts)), columns=['Coins', 'ARIMA+GARCH']))
    rsi_predictions = calculate_RSI(prices, 15)
    print(pd.DataFrame(list(zip(coins, rsi_predictions)), columns=['Coins', 'RSI']))
    ama_predictions = calculate_AMA(prices, 10, 100)
    print(pd.DataFrame(list(zip(coins, ama_predictions)), columns=['Coins', 'AMA']))
    act = ["" for i in range(len(coins))]
    for i in range(len(coins)):
        if portfo[i] == 0:
            act[i] = "N/A"
            continue
        if (ama_predictions[i] == 1 or arga_predicts[i] == 1) or rsi_predictions[i] == 1:
            act[i] = "BUY"
        else:
            act[i] = "SELL"
    print(pd.DataFrame(list(zip(coins, act)), columns=['Coins', 'Actions']))
    return act


def tactics_bear(prices, portfo):
    arga_predicts = arima_garch_prediction(prices, portfo, 120)
    print(pd.DataFrame(list(zip(coins, arga_predicts)), columns=['Coins', 'ARIMA+GARCH']))
    rsi_predictions = calculate_RSI_bear(prices, 15)
    print(pd.DataFrame(list(zip(coins, rsi_predictions)), columns=['Coins', 'RSI']))
    ama_predictions = calculate_AMA(prices, 10, 100)
    print(pd.DataFrame(list(zip(coins, ama_predictions)), columns=['Coins', 'AMA']))
    act = ["" for i in range(len(coins))]
    for i in range(len(coins)):
        if portfo[i] == 0:
            act[i] = "N/A"
            continue
        if (ama_predictions[i] == 1 or arga_predicts[i] == 1) or rsi_predictions[i] == 1:
            act[i] = "BUY"
        else:
            act[i] = "SELL"
    print(pd.DataFrame(list(zip(coins, act)), columns=['Coins', 'Actions']))
    return act


def strategy(prices_3m, prices_6m):
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


def get_ratio(prices_3m, prices_6m, weights):
    tday_returns_3m = prices_3m.pct_change()
    tday_returns_6m = prices_6m.pct_change()
    tday_returns_poz_3m = replace_negatives(prices_3m.pct_change())
    tmean_sharpe_3m = tday_returns_3m.mean()
    tmean_sharpe_6m = tday_returns_6m.mean()
    tmean_sortino_3m = tday_returns_poz_3m.mean()
    tsharpe_3m_cov = tday_returns_3m.cov()
    tsharpe_6m_cov = tday_returns_6m.cov()
    tsortino_3m_cov = tday_returns_poz_3m.cov()
    ratios = calculate_ratios(tmean_sharpe_3m, tmean_sharpe_6m, tmean_sortino_3m, tsharpe_3m_cov, tsharpe_6m_cov,
                              tsortino_3m_cov, weights)
    return ratios

def get_sharpe(prices_3m, weights):
    tday_returns_3m = prices_3m.pct_change()
    tmean_sharpe_3m = tday_returns_3m.mean()
    tsharpe_3m_cov = tday_returns_3m.cov()
    ratios = calculate_sharpe(tmean_sharpe_3m, tsharpe_3m_cov, weights, len(prices_3m))
    return ratios

def calculate_sharpe(mean_return_3m, cov_matrix_3m, weights_coins, len):
    returns_3m = np.sum(mean_return_3m * weights_coins) * len
    std_dev_3m = np.sqrt(np.dot(weights_coins.T, np.dot(cov_matrix_3m, weights_coins))) * np.sqrt(len)
    risk_free_return = 0.012
    sharpe_3m = (returns_3m - risk_free_return) / std_dev_3m
    rez = sharpe_3m
    return rez


yf.pdr_override()
coins = ['ADA-USD', 'BCH-USD', 'BNB-USD', 'BTC-USD', 'DOGE-USD', 'ETH-USD', 'KMD-USD', 'LTC-USD', 'XLM-USD',
         'XRP-USD']
# teže koliko bomo upoštevali kater indikator v optimizaciji format [SHARPE(3m), SHARPE(6m), SORTINO(3m)]
weights_ratios = [0.4, 0.3, 0.3]
datum = date(2021, 8, 3)
start_date_test = datum + relativedelta(days=-729)
end_date_test = datum + relativedelta(days=-1)
# test_1day_data = get_closing(coins, start_date_test + relativedelta(months=-6), end_date_test)
# test_1day_data.to_csv("1daydata.csv", encoding='utf-8')
test_1day_data = pd.read_csv("1daydata.csv", parse_dates=[0], index_col=0)
# test_4h_data = get_4h_data(coins, start_date_test, end_date_test).fillna(method='ffill')
# test_4h_data.to_csv("4hdata.csv", encoding='utf-8')
test_4h_data = pd.read_csv("4hdata.csv", parse_dates=[0], index_col=0).fillna(method='ffill')
today = start_date_test + relativedelta(months=+3)
pct_profit = [[] for i in range(len(coins))]
loss = 0
win = 0
big_win = 0
big_loss = 0
loss_opt = 0
win_opt = 0
big_win_opt = 0
big_loss_opt = 0
st_akcij = 0
money = 100
money_opt = 100
wallets = np.zeros(len(coins))
wallets_opt = np.zeros(len(coins))
profits = np.zeros(len(coins))
positions = np.zeros(len(coins))
buying_prices = np.zeros(len(coins))
test_port = np.zeros(len(coins))
btc_port = np.zeros(len(coins))
be_port = np.zeros(len(coins))
btc_port[3] = 1
be_port[3] = 0.5
be_port[5] = 0.5
today_plus_3m = today
bear = False
btc_prices = []
eth_prices = []
wallet_sum = []
opt_only_sum = []
sharpe_wal = []
sharpe_btc = []
sharpe_be = []
positions_opt_only = np.zeros(len(coins))
buying_prices_opt_only = np.zeros(len(coins))
while today < end_date_test:
    bad = False
    if today >= today_plus_3m:
        st_akcij = 0
        print("Strategy change" + "date: " + str(today))
        wallets = np.zeros(len(coins))
        wallets_opt = np.zeros(len(coins))
        test_port = strategy(test_1day_data[(today + relativedelta(months=-3)):today],
                             test_1day_data[(today + relativedelta(months=-6)):today])
        today_plus_3m = today + relativedelta(months=+3)
        ratio = get_ratio(test_1day_data[(today + relativedelta(months=-3)):today],
                          test_1day_data[(today + relativedelta(months=-6)):today], test_port)
        for i in range(0, len(coins)):
            if test_port[i] == 0:
                continue
            else:
                wallets[i] = money * test_port[i]
                wallets_opt[i] = money_opt * test_port[i]
        if ratio > 2:
            bear = False
        else:
            bear = True
    if np.array_equal(test_port, np.zeros(len(coins))) and not bad:
        # today = today + relativedelta(months=+3)
        bad = True
        print("Bad strategy, skip 3 months")
    # print(test_4h_data[:today])
    t2 = datetime.combine(today, datetime.min.time())
    t2 = pytz.utc.localize(t2)
    prices_4h = test_4h_data[:t2 + relativedelta(days=+1)]
    print(t2)
    for i in range(len(coins)):
        if test_port[i] != 0 and positions_opt_only[i] == 0:
            positions_opt_only[i] = 1
            buying_prices_opt_only[i] = prices_4h.tail(7)[coins[i]].iloc[0]
            print("Buy (optimisation only) " + coins[i])
    for ix in range(-6, 0):
        cur_prices = prices_4h[:ix]
        btc_prices.append(float(cur_prices.tail(1)[coins[3]].iloc[0]))
        eth_prices.append(float(cur_prices.tail(1)[coins[5]].iloc[0]))
        sharpe_wal.append(get_sharpe(cur_prices[t2 + relativedelta(months=-3):], test_port))
        sharpe_btc.append(get_sharpe(cur_prices[t2 + relativedelta(months=-3):], btc_port))
        sharpe_be.append(get_sharpe(cur_prices[t2 + relativedelta(months=-3):], be_port))
        sum_assets = 0
        # getting prices for optimisation only
        for sumdex in range(len(coins)):
            if test_port[sumdex] != 0:
                if bool(positions_opt_only[sumdex] == 1):
                    amount = wallets_opt[sumdex] / buying_prices_opt_only[sumdex]
                    pricediff = amount * (cur_prices.tail(1)[coins[sumdex]].iloc[0])
                    sum_assets = sum_assets + pricediff
                else:
                    sum_assets = sum_assets + wallets[sumdex]
            if np.array_equal(test_port, np.zeros(len(coins))):
                sum_assets = money_opt
        opt_only_sum.append(float(sum_assets))
        sum_assets = 0
        # getting prices for the whole strategy
        for sumdex in range(len(coins)):
            if test_port[sumdex] != 0:
                if bool(positions[sumdex] == 1):
                    amount = wallets[sumdex] / buying_prices[sumdex]
                    pricediff = amount * (cur_prices.tail(1)[coins[sumdex]].iloc[0])
                    sum_assets = sum_assets + pricediff
                else:
                    sum_assets = sum_assets + wallets[sumdex]
            if np.array_equal(test_port, np.zeros(len(coins))):
                sum_assets = money
        wallet_sum.append(float(sum_assets))
        print(wallets)
        print(sum_assets)
        if not bad:
            if bear:
                actions = tactics_bear(cur_prices, test_port)
            else:
                actions = tactics(cur_prices, test_port)
        for i in range(0, len(coins)):
            cur_price = cur_prices.tail(1)[coins[i]].iloc[0]
            if test_port[i] == 0:
                continue
            # Varovalni mehanizem
            if bool(cur_price < buying_prices[i]) and bool(positions[i] == 1):
                st_akcij += 1
                positions[i] = 0
                amount = wallets[i] / buying_prices[i]
                pricediff = amount * (cur_price - buying_prices[i])
                pricediff = pricediff - (0.0002 * pricediff)
                profits[i] += pricediff.item()
                if bool(pricediff.item() > 0):
                    win += 1
                    if bool(pricediff.item() > big_win):
                        big_win = pricediff.item()
                if bool(pricediff.item() < 0):
                    loss += 1
                    if bool(pricediff.item() < big_loss):
                        big_loss = pricediff.item()
                pctdiff = (cur_price / buying_prices[i]) - 1
                pct_profit[i].append(pctdiff * test_port[i])
                wallets[i] = amount * cur_price
                print("Panic sell " + coins[i] + " diff: " + str(pricediff) + " pct: " + str(pctdiff) +
                      " Profits: " + str(profits[i]) + " wallet: " + str(wallets[i]))
                continue
            if bool(actions[i] == "BUY") and bool(positions[i] == 0):
                st_akcij += 1
                positions[i] = 1
                buying_prices[i] = cur_price
                print("Buy " + coins[i])
                continue
            if bool(actions[i] == "SELL") and bool(positions[i] == 1):
                st_akcij += 1
                positions[i] = 0
                amount = wallets[i] / buying_prices[i]
                pricediff = amount * (cur_price - buying_prices[i])
                pricediff = pricediff - (0.0002 * pricediff)
                profits[i] += pricediff.item()
                if bool(pricediff.item() > 0):
                    win += 1
                    if bool(pricediff.item() > big_win):
                        big_win = pricediff.item()
                if bool(pricediff.item() < 0):
                    loss += 1
                    if bool(pricediff.item() < big_loss):
                        big_loss = pricediff.item()
                pctdiff = (cur_price / buying_prices[i]) - 1
                pct_profit[i].append(pctdiff * test_port[i])
                wallets[i] = amount * cur_price
                print("Sell " + coins[i] + " diff: " + str(pricediff) + " pct: " + str(pctdiff) +
                      " Profits: " + str(profits[i]) + " wallet: " + str(wallets[i]))
                continue
    today = today + relativedelta(days=+1)
    if today >= today_plus_3m or today >= end_date_test:
        t2 = datetime.combine(today, datetime.min.time())
        t2 = pytz.utc.localize(t2)
        cur_prices = test_4h_data[:t2 + relativedelta(days=+1)]
        actions = tactics(cur_prices, test_port)
        #sell at period end
        for i in range(0, len(coins)):
            cur_price = cur_prices.tail(1)[coins[i]]
            if test_port[i] == 0:
                continue
            if positions[i] == 1:
                st_akcij += 1
                positions[i] = 0
                amount = wallets[i] / buying_prices[i]
                pricediff = amount * (cur_price - buying_prices[i])
                pricediff = pricediff - 0.0002 * pricediff
                profits[i] += pricediff
                if bool(pricediff.item() > 0):
                    win += 1
                    if bool(pricediff.item() > big_win):
                        big_win = pricediff.item()
                if bool(pricediff.item() < 0):
                    loss += 1
                    if bool(pricediff.item() < big_loss):
                        big_loss = pricediff.item()
                pctdiff = (cur_price / buying_prices[i]) - 1
                pct_profit[i].append(pctdiff * test_port[i])
                wallets[i] = amount * cur_price
                print("Sell all " + coins[i] + " diff: " + str(pricediff) + " pct: " + str(pctdiff) +
                      " Profits: " + str(profits[i]) + " wallet: " + str(wallets[i]))
        # opt only sell at period end
        for i in range(0, len(coins)):
            cur_price = cur_prices.tail(1)[coins[i]]
            if test_port[i] == 0:
                continue
            if positions_opt_only[i] == 1:
                positions_opt_only[i] = 0
                amount = wallets_opt[i] / buying_prices_opt_only[i]
                pricediff = amount * (cur_price - buying_prices_opt_only[i])
                pricediff = pricediff - 0.0002 * pricediff
                if bool(pricediff.item() > 0):
                    win_opt += 1
                    if bool(pricediff.item() > big_win_opt):
                        big_win_opt = pricediff.item()
                if bool(pricediff.item() < 0):
                    loss_opt += 1
                    if bool(pricediff.item() < big_loss_opt):
                        big_loss_opt = pricediff.item()
                wallets_opt[i] = amount * cur_price
                print("Sell all " + coins[i] + " diff: " + str(pricediff)
                      + " wallet: " + str(wallets_opt[i]))
        print("Število akcij : "+str(st_akcij))
        wallet_sum.append(float(0))
        opt_only_sum.append(float(0))
        if not bad:
            money = sum(wallets)
            money_opt = sum(wallets_opt)
print(sum(profits))
bt = open("rezultati/btcPrice.txt", "w")
bt.write(str(btc_prices))
bt.close()
et = open("rezultati/ethPrice.txt", "w")
et.write(str(eth_prices))
et.close()
bt = open("rezultati/btcSharpe.txt", "w")
bt.write(str(sharpe_btc))
bt.close()
et = open("rezultati/beSharpe.txt", "w")
et.write(str(sharpe_be))
et.close()
wal = open("rezultati/walletSharpe.txt", "w")
wal.write(str(sharpe_wal))
wal.close()
wal = open("rezultati/wallet.txt", "w")
wal.write(str(wallet_sum))
wal.close()
wal2 = open("rezultati/wallet_opt.txt", "w")
wal2.write(str(opt_only_sum))
wal2.close()
f = open("rezultati/winloss.txt", "w")
f.write("Wins: " + str(win) + "\nLoss: " + str(loss) + "\nBigwin: " + str(big_win) + "\nBigloss: " + str(big_loss))
f.close()
f = open("rezultati/winloss_opt.txt", "w")
f.write("Wins: " + str(win_opt) + "\nLoss: " + str(loss_opt) + "\nBigwin: " + str(big_win_opt) + "\nBigloss: " + str(
    big_loss_opt))
f.close()
