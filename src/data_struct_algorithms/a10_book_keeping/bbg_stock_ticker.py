

# https://leetcode.com/discuss/interview-question/algorithms/124837/stock-ticker
# similar to leadership board
import heapq
class StockSticker:
    def __init__(self, k: int):
        self.top_size = k
        self.prices = {}
        self.heap = []

    def add_update(self, stock: str, price: float):
        self.prices[stock] = price
        heapq.heappush(self.heap, (price, stock))
        if len(self.heap) > self.top_size:
            heapq.heappop(self.heap)

    def top(self): # O(k)
        return {s: p for p, s in self.heap}
