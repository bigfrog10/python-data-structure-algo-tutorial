import itertools
import bisect


def price_prctl(price_txs: list, prctl):
    # list of (tx, price) history
    price_txs.sort(key=lambda x: x[0])  # sort by prices, O(nlogn)

    tx_accumu = list(itertools.accumulate(x[1] for x in price_txs))  # O(n)

    idx = bisect.bisect_left(tx_accumu, tx_accumu[-1] * prctl)  # -1 has total, O(logn)

    return price_txs[idx][0]


pt = [[999, 345], [1250, 515], [595, 7000]]  # 595
print(price_prctl(pt, 0.8))

pt = [[999, 345], [1250, 5150], [595, 700]]  # 1250
print(price_prctl(pt, 0.8))

pt = [[999, 3450], [1250, 515], [595, 700]]  # 999
print(price_prctl(pt, 0.8))
