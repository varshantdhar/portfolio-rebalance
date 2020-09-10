# Dynamic Portfolio Transition

## Problem


This problem asks you to consider a rebalancing decision in the context of a taxable account, where a client is subject to taxation on realized capital gains. The goal is to implement a tax-efficient portfolio rebalancing scheme.

The client made three separate deposits of $100k at the following dates 
(2010-01-04, 2014-01-02, 2018-01-02), on each day purchasing an equal-weighted combination of the of five ETFs: VTI,VEA,VWO,BND and EMB. Suppose the client would like to rebalance the portfolio in a tax-efficient manner at the close of business on Mar. 14, 2018. 

Additional information:
(1) If the client were starting from cash, she would again select an equal-weighted portfolio.
(2) The client faces a short-term cap gains rate of 30% and long term cap gains tax of 20%.
(3) The forward looking variance-covariance matrix of the assets is given by the sample covariance matrix.

Questions:
1. Write down an objective function for the tax-efficient migration problem. This objective function should articulate the tradeoff between the investment efficiency of the allocation and the tax cost of the transition. Describe the intuition.
2. Implement the optimization problem in code. What allocation would the client trade to?
3. Would the client’s remaining investment horizon impact the allocation the client will trade to? For example, consider an investor who intends to rebalance and then remain invested for only one year vs. an investor who intends to rebalance and then remain invested for another five years.

## Program

```shell
python3 portfolio_optim.py
python3 portfolio_optim.py --inv_horizon=1
```

The first command runs the optimization program for a 5-year horizon by default and the second command changes that to a 1-year horizon using the argument ```--inv_horizon```.

