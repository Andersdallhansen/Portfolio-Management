from baseclass import PORTFOLIO
import numpy as np


class FTL(PORTFOLIO):
    def __init__(self, N):
        self.N = N
        self.reset()

    def reset(self):
        self.cum_returns = None
        self.reset_portfolio()

    def rebalance(self, returns):
        if returns is not None:
            if self.cum_returns is None:
                self.cum_returns = returns
            else:
                self.cum_returns = self.cum_returns * returns

            leader = np.argmax(self.cum_returns)
            new_allocation = np.zeros(self.N + 1)
            new_allocation[leader] = 1
            self.rebalance_portfolio(new_allocation)

    def step(self, returns):
        self.step_portfolio(returns)


class UBAH(PORTFOLIO):
    def __init__(self, N):
        self.N = N
        self.reset()
        self.rebalance([0.] + [1 / self.N] * self.N)

    def reset(self):
        self.reset_portfolio()

    def rebalance(self, returns):
        pass

    def step(self, returns):
        self.step_portfolio(returns)


class UCRP(PORTFOLIO):
    def __init__(self, N):
        self.N = N
        self.reset()
        self.rebalance()

    def reset(self):
        self.reset_portfolio()

    def rebalance(self):
        self.rebalance([0.] + [1 / self.N] * self.N)

    def step(self, returns):
        self.step_portfolio(returns)