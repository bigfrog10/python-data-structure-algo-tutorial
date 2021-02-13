###############################################################################
# Say you have an array prices for which the ith element is the price of a
# given stock on day i.
#
# Design an algorithm to find the maximum profit. You may complete as many
# transactions as you like (i.e., buy one and sell one share of the stock
# multiple times).
#
# Note: You may not engage in multiple transactions at the same time (i.e., you
# must sell the stock before you buy again).
# EASY
################################################################################

def max_gain(prices: list) -> float:
    profit = 0
    bp = 0  # buy price
    open_tx = False  # mark open transaction or not, i.e., there is a buy or not.
    for i in range(len(prices) - 1):
        if prices[i] < prices[i+1] and not open_tx:
            bp = prices[i]
            open_tx = True
        if prices[i] > prices[i+1] and open_tx:
            profit += prices[i] - bp
            open_tx = False

    if open_tx:
        profit += prices[-1] - bp
    return profit


print(max_gain([7, 1, 5, 3, 6, 4]))
print(max_gain([1, 2, 3, 4, 5]))
print(max_gain([5, 4, 3, 2, 1]))
