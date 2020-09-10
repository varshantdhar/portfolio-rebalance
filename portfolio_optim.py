# Portfolio Optimization Problem
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import argparse

from scipy.optimize import minimize

'''
Helper function for load_data() that builds the portfolio variable using the 
historical returns and cashflow insertions specified in the problem.

Args:
    basic (pandas series): historical returns dataset for ETFs
    portfolio (list of dicts): historical returns organized into a list of dicts
    cashflow (list of dt.datetime): list of dates for cashflow insertsions, based
        on problem.
Returns:
    portfolio (list of dicts): gives the market value of the assets for each ETF
        on each day based on the cash invested.
'''
def build_portfolio(basic, portfolio, cashflow):
    for idx, row in basic.iterrows():
        portfolio[idx]["Date"] = row["Date"].date()
        keys = list(basic.columns[1:])
        if row["Date"] in cashflow:
            for key in keys:
                portfolio[idx][key] = np.append(portfolio[idx][key], [20000.0])
        for key in keys:
            growth = row[key]
            portfolio[idx][key] = np.array(
                list(map(lambda x: x + growth * x, portfolio[idx][key]))
            )
        portfolio.append(portfolio[idx].copy())
    return portfolio
'''
Helper function for load_data() that finds the annual mean and standard deviation
for the dataset of historical returns. 

Args:
    portfolio (list of dicts): gives the market value of the assets for each ETF
        on each day based on the cash invested.
    basic (pandas series): returns dataset for ETFs
Returns:
    mu (numpy array): average annual returns for each ETF
    std (numpy array): standard deviation for annual returns for each ETF
'''
def mean_std(portfolio, basic):
    times = [i for i in range(0, len(portfolio), 253)]
    year_returns = []
    for time in times:
        year_returns.append(basic.iloc[time : time + 1, 1:].sum() * 100)
    mu = np.mean(year_returns, axis=0)
    std = np.std(year_returns, axis=0)
    return (mu, std)

'''
Loads the data from the dataset. 

Returns:
    portfolio (list of dicts): gives the market value of the assets for each ETF
        on each day based on the cash invested.
    mu (numpy array): average annual returns for each ETF
    std (numpy array): standard deviation for annual returns for each ETF
    cov (numpy array): covariance matrix of returns for each ETF
    basic (pandas series): returns dataset for ETFs
'''
def load_data():
    returns = (
        pd.read_csv("HistoricalReturns.csv")
        .rename(columns={"Unnamed: 0": "Date"})
        .dropna()
    )
    returns["Date"] = pd.to_datetime(returns["Date"])
    basic = returns.iloc[:, :6]
    cov = np.array(np.dot(np.transpose(np.array(basic.values[:, 1:])), np.array(basic.values[:, 1:])), dtype="float")
    portfolio = []
    portfolio.append(basic.iloc[0,].to_dict())
    keys = list(basic.columns[1:])
    for key in keys:
        portfolio[0][key] = np.array([20000.0])
    cashflow = [dt.datetime(2014, 1, 2), dt.datetime(2018, 1, 2)]
    portfolio = build_portfolio(basic, portfolio, cashflow)
    mu, std = mean_std(portfolio, basic)
    return (portfolio, mu, std, cov, basic)

'''
Finds the amount of tax incurred, serves as a helper function to tax_optim() which is 
optimized to find the minimum value

Args:
    alloc (list of int): initial list of allocation of an ETF for all cash insertions. 
        Equal-weighted.
    final_alloc (list of int): final allocation of an ETF for all cash insertions
    amount (int): amount needed for re-allocation
    rate (numpy array): rate of gains/losses for each cash insertion / investment.
Returns:
    val (int): amount of tax incurred for the desired allocation
'''
def sell_alloc(alloc, final_alloc, amount, rate):
    val = 0
    tax = [0.2, 0.2, 0.3]
    for i in range(len(final_alloc)):
        val += (alloc[i] - alloc[i] / (1 + rate[i])) * tax[i]
    return val

'''
Returns the amount of tax applied based on capital gains (long and short). Minimizes
the taxable amount by minimally selling assets with higher gains and selling more assets
with loss. This function serves as a helper function to finding the optimal portfolio.

Args:
    portfolio (list of dicts): gives the market value of the assets for each ETF
        on each day based on the cash invested.
    init_wt (numpy array): percentage of allocation to ETFs.
Returns:
    max(0.0,val): maximum between 0 and the value of capital gains tax calculated
        based on the specified allocation
'''
def tax_optim(portfolio, init_wt):
    val = 0
    final_alloc = pd.DataFrame(portfolio[len(portfolio) - 1]).values[:, 1:]
    total_alloc = final_alloc.sum(axis=0)
    tot_sum = np.sum(total_alloc)
    new_alloc = tot_sum * init_wt
    diff = new_alloc - total_alloc
    sells = [i for i, val in enumerate(diff < 0) if val]
    for s in sells:
        arr = final_alloc[:, s]
        rate = (arr - 20000) / 20000
        bounds = ((0, arr[0]), (0, arr[1]), (0, arr[2]))
        amount = -diff[s]
        alloc = [amount / len(arr)] * len(arr)
        cons = {"type": "eq", "fun": lambda x: np.sum(x) - amount}
        n_alloc = minimize(
            sell_alloc, alloc, args=(arr, amount, rate), bounds=bounds, constraints=cons
        )
        final_alloc[:, s] = final_alloc[:, s] - n_alloc.x
        val += n_alloc.fun
    return max(0.0,val)

'''
Calculates the Markowitz frontier.

Args:
    init_wt (numpy array): initial percentage of allocation to ETFs.
    mu (numpy array): average annual returns for each ETF
    cov (numpy array): covariance matrix of returns for each ETF
    portfolio (list of dicts): gives the market value of the assets for each ETF
        on each day based on the cash invested.
Returns:
    total (int): Markowitz total frontier value
'''
def portfolio_frontier(init_wt, mu, cov, coeff, portfolio):
    val = (
        np.dot(np.dot(np.transpose(init_wt), cov), init_wt)
        - np.sum(np.multiply(init_wt, np.diag(cov)))
         * np.transpose(mu) * init_wt
    )
    taxes = tax_optim(portfolio, init_wt)
    total = coeff * np.sum(val) + taxes
    return total

'''
Simulation of portfolio allocation using GBMs. Calculates the average gain in 
revenue and decrease in risk using a 1000 simulations. 

Args:
    mu (numpy array): average annual returns for each ETF
    std (numpy array): standard deviation for annual returns for each ETF
    fin (numpy array): asset allocation for each ETF
    optim (numpy array): optimum asset allocation for each ETF
    basic (pandas series): returns dataset for ETFs
    years (int): investment horizon
'''
def simulation(mu, std, fin, optim, basic, years):
    risk = []
    profit = []
    for sim in range(1000):
        _fin = fin.copy()
        _optim = optim.copy()
        returns = np.array(basic.values[:, 1:], dtype=float)
        r_mu = np.mean(returns,axis=0)
        r_std = np.std(returns,axis=0)
        for idx in range(253 * years): 
            returns = np.vstack(
                (
                    returns,
                    np.array(
                        1.0
                        - np.exp((r_mu - (r_std ** 2 / 2)) + r_std * np.random.normal(r_mu, r_std))
                    ),
                )
            )
            r_mu = np.mean(returns,axis=0)
            r_std = np.std(returns,axis=0)
        times = [i for i in range(0, len(returns), 253)]
        year_returns = []
        for time in times:
            year_returns.append(np.sum(returns[time : time + 1, :], axis=0) * 100)
        mu = np.mean(year_returns, axis=0)
        std = np.std(year_returns, axis=0)
        final_alloc = fin/np.sum(fin)
        risk.append(
            {
                "optim": np.dot(optimizer.x, np.transpose(std)),
                "fin": np.dot(final_alloc, np.transpose(std)),
            }
        )
        for idx in range(len(returns[2063:,])):
            _fin = _fin + np.multiply(_fin, returns[2063 + idx,])
            _optim = _optim + np.multiply(_optim, returns[2063 + idx,])
        profit.append({"optim": np.sum(_optim), "fin": np.sum(_fin)})
    optim_p = sum(p["optim"] for p in profit) / len(profit)
    optim_r = sum(r["optim"] for r in risk) / len(risk)
    fin_p = sum(p["fin"] for p in profit) / len(profit)
    fin_r = sum(r["fin"] for r in risk) / len(risk)
    print("Average gain in revenue (1000 Simulations): $" +
        str(optim_p - fin_p))
    print("Decrease in risk (1000 Simulations): " + 
        str(-((optim_r - fin_r)/optim_r) * 100) + " %"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inv_horizon', dest='years', type=int, nargs=1,
        help='number of years of investment horizon', default=5)
    args = parser.parse_args()
    portfolio, mu, std, cov, basic = load_data()
    final_alloc = np.array(
        pd.DataFrame(portfolio[len(portfolio) - 1]).iloc[:, 1:].sum()
    )
    final_portfolio = np.sum(final_alloc)
    init_wt = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    fin = init_wt * (final_portfolio - tax_optim(portfolio, init_wt))
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
    risk = np.sum(np.multiply(init_wt, np.diag(cov)))
    coeff = tax_optim(portfolio,init_wt) / np.sum(
        np.dot(np.dot(np.transpose(init_wt), cov), init_wt)
        -  risk * np.transpose(mu) * init_wt
    )
    optimizer = minimize(
        portfolio_frontier,
        init_wt,
        args=(mu, cov, coeff, portfolio),
        constraints=cons,
        bounds=bounds,
    )
    optim = optimizer.x * (final_portfolio - tax_optim(portfolio, optimizer.x))
    simulation(mu, std, fin, optim, basic, args.years)



