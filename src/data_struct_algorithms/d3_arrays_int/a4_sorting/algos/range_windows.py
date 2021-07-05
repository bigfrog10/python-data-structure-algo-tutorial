import bisect

ranges = [(2, 5), (6, 10), (11, 16), (17, 100)]

def fit(prices: list):
    windows = {}
    for symbol, price in prices:
        idx = bisect.bisect(ranges, (price, price))
        windows[symbol] = idx - 1

    return windows

prices = [('APPL', 5), ('APPL', 7), ('APPL', 12)]
print(fit(prices))
