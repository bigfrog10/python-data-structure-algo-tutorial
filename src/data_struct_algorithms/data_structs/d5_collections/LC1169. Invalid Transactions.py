# LC1169. Invalid Transactions
import bisect
from collections import Counter, defaultdict
from typing import List
class Solution:
    def invalidTransactions(self, transactions: List[str]) -> List[str]:
        counter = Counter(transactions)
        tx_cache = defaultdict(list)
        ret = set()
        for tx in transactions:
            name, ts, amount, city = tx.split(',')
            invalid = False
            if int(amount) > 1000: invalid = True
            hist = tx_cache[name]
            ts = int(ts)
            idx = bisect.bisect_left(hist, (ts, city, tx))
            i = idx-1
            while i >= 0 and ts - hist[i][0] <= 60:
                if hist[i][1] != city:
                    ret.add(hist[i][2])
                    invalid = True
                i -= 1
            i = idx
            while i < len(hist) and hist[i][0] - ts <= 60:
                if hist[i][1] != city:
                    ret.add(hist[i][2])
                    invalid = True
                i += 1
            if invalid: ret.add(tx)
            hist.insert(idx, (ts, city, tx))
            #bisect.insort(hist, (ts, city, tx))
        ret1 = []
        for tx in ret: ret1.extend([tx] * counter[tx])
        return ret1
