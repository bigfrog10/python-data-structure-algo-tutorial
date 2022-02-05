# LC158. Read N Characters Given Read4 II - Call multiple times
class Solution:
    def __init__(self): self.cache = []
    def read(self, buf: List[str], n: int) -> int:
        filled = 0
        while filled < n:
            b = [''] * 4
            count = read4(b)
            self.cache.extend(b[:count])

            needed = min(n-filled, len(self.cache))
            if needed == 0: break # no more read

            buf[filled:] = self.cache[:needed]
            self.cache = self.cache[needed:]
            filled += needed
        return filled

# LC157. Read N Characters Given Read4
def read(self, buf, n):
    size = 0
    while size < n:
        b = [''] * 4
        count = read4(b)
        if count == 0: break
        bs = min(n - size, count)
        buf[size:] = b[:bs]
        size += count
    return min(size, n)

# LC346. Moving Average from Data Stream
class MovingAverage:
    def __init__(self, size: int):
        self.size = size
        self.queue = deque()
        self.mv = 0
    def next(self, val: int) -> float:
        size, queue = self.size, self.queue
        queue.append(val)
        if len(queue) > size:
            v = queue.popleft()
            self.mv = self.mv + (val - v) / size
        else: self.mv = sum(queue) / len(queue)
        return self.mv
