################################################################################
# Say you have an array for which the ith element is the price of a given stock
# on day i.
#
# If you were only permitted to complete at most one transaction (i.e., buy one
# and sell one share of the stock), design an algorithm to find the maximum
# profit.
#
# Note that you cannot sell a stock before you buy one.
# EASY
################################################################################
# O(n^2)
def max_gain1(prices: list) -> float:
    ret = 0
    for i, p in enumerate(prices):
        for j, q in enumerate(prices[i+1:]):
            if q - p > ret:
                ret = q - p

    return ret


import sys


# Kadane's Algorithm
def max_gain(prices: list) -> float:
    # instead of maintaining 2 boundaries, we maintain the following
    profit = 0
    min_price = sys.maxsize
    for i, p in enumerate(prices):
        if p < min_price:
            min_price = p
        elif p - min_price > profit:
            profit = p - min_price

    return profit


print(max_gain([7, 1, 5, 3, 6, 4]))
print(max_gain([7, 6, 4, 3, 1]))
print(max_gain([2, 1, 2, 0, 1]))
