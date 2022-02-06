
# LC1570. Dot Product of Two Sparse Vectors
class SparseVector:
    def __init__(self, nums: List[int]):
        self.sparse = {}
        for i, n in enumerate(nums):
            if n != 0: self.sparse[i] = n
    def dotProduct(self, vec: 'SparseVector') -> int:
        dot = 0
        for i, n in self.sparse.items():
            if i in vec.sparse: dot += n * vec.sparse[i]
        return dot

# LC380. Insert Delete GetRandom O(1), RandomizedSet, top100
import random
class RandomizedSet:
    def __init__(self):
        self.values = []
        self.index = {}  # value 2 positions
    def insert(self, val: int) -> bool: # O(1)
        if val in self.index: return False
        self.values.append(val)
        self.index[val] = len(self.values) - 1
        return True
    def remove(self, val: int) -> bool:  # O(1)
        if val not in self.index: return False
        pos = self.index[val]
        self.index[self.values[-1]] = self.index[val]
        del self.index[val]  # handle index
        self.values[pos] = self.values[-1]
        self.values.pop()  # swap to last and remove
        return True
    def getRandom(self) -> int:  # O(1), just values[random.randrange(len(values))]
        return random.choice(self.values)

# LC381. Insert Delete GetRandom O(1) - Duplicates allowed
class RandomizedCollection: # 93%, fast
    def __init__(self):
        self.values = []
        self.idxs = defaultdict(set)  # value to indices
    def insert(self, val: int) -> bool:
        exist = val in self.idxs
        self.values.append(val)
        self.idxs[val].add(len(self.values)-1)
        return not exist
    def remove(self, val: int) -> bool:
        if val not in self.idxs: return False
        # we need to be careful when there is only 1 element, this order is important.
        idx = self.idxs[val].pop()  # to be removed
        last = self.values[-1]  # remove len - 1 for last element
        self.values[idx] = last  # move last value to the removed spot
        self.idxs[last].add(idx)
        self.idxs[last].remove(len(self.values)-1)  # remove last idx, it is len - 1
        # this has to be done after last's index is updated. Otherwise, fail with 1 element.
        if len(self.idxs[val]) == 0: del self.idxs[val]
        self.values.pop()  # remove last element
        return True
    def getRandom(self) -> int:
        return random.choice(self.values)

# LC295. Find Median from Data Stream, top100
class MedianFinder:
    def __init__(self):
        self.heaps = [], []  # max heap from smalls, min heap from larges
    def addNum(self, num: int) -> None:
        small, large = self.heaps
        heappush(small, -heappushpop(large, num))
        if len(large) < len(small): heappush(large, -heappop(small))
    def findMedian(self) -> float:
        small, large = self.heaps
        if len(large) > len(small): return float(large[0])
        return (large[0] - small[0]) / 2.0

# LC703. Kth Largest Element in a Stream
class KthLargest:
    def __init__(self, k, nums):
        self.nums = nums
        self.k = k
        heapq.heapify(self.nums)  # min heap
        while len(self.nums) > k: heapq.heappop(self.nums) # min popped
    def add(self, val):
        if len(self.nums) < self.k: heapq.heappush(self.nums, val)
        elif val > self.nums[0]: heapq.heapreplace(self.nums, val)
        return self.nums[0]  # min, which is the kth largest

# LC535. Encode and Decode TinyURL
from random import choices
chars = string.ascii_letters + string.digits  # upper, lower, digits
class Codec:
    def __init__(self):
        self.url2code = {}
        self.code2url = {}
    def encode(self, longUrl: str) -> str:  # Encodes a URL to a shortened URL.
        while longUrl not in self.url2code:
            code = ''.join(choices(chars, k=7))
            if code not in self.code2url:
                self.code2url[code] = longUrl
                self.url2code[longUrl] = code
        return 'http://tinyurl.com/' + self.url2code[longUrl]
    def decode(self, shortUrl: str) -> str:  # Decodes a shortened URL to its original URL.
        return self.code2url[shortUrl[-7:]]

# LC641. Design Circular Deque
class MyCircularDeque:
    def __init__(self, k: int):
        self.k = k
        self.arr = [-1] * k  # -1 is required by the problem
        self.front = 0 # current empty
        self.rear = 1 # current empty
        self._size = 0
    def insertFront(self, value: int) -> bool:
        if self.isFull(): return False
        self.arr[self.front] = value
        self.front = (self.front - 1) % self.k
        self._size += 1
        return True
    def insertLast(self, value: int) -> bool:
        if self.isFull(): return False
        self.arr[self.rear] = value
        self.rear = (self.rear + 1) % self.k
        self._size += 1
        return True
    def deleteFront(self) -> bool:
        if self.isEmpty(): return False
        self.front = (self.front + 1) % self.k
        self.arr[self.front] = -1 # -1 is required by the problem
        self._size -= 1
        return True
    def deleteLast(self) -> bool:
        if self.isEmpty(): return False
        self.rear = (self.rear - 1) % self.k
        self.arr[self.rear] = -1 # -1 is required by the problem
        self._size -= 1
        return True
    def getFront(self) -> int:
        f = (self.front + 1) % self.k
        return self.arr[f]
    def getRear(self) -> int:
        r = (self.rear - 1 + self.k) % self.k
        return self.arr[r]
    def isEmpty(self) -> bool: return self._size == 0
    def isFull(self) -> bool: return self._size == self.k
# LC1244. Design A Leaderboard
class Leaderboard:
    def __init__(self):
        self.A = collections.Counter()
    def addScore(self, playerId: int, score: int) -> None:
        self.A[playerId] += score
    def top(self, K: int) -> int:
        return sum(v for i,v in self.A.most_common(K))  # roughly nlogk + klogk
    def reset(self, playerId: int) -> None:
        self.A[playerId] = 0

# LC1032. Stream of Characters
class StreamChecker:
    def __init__(self, words: List[str]):
        self.trie = {}
        for word in set(words): # remove duplicate words
            node = self.trie
            for ch in word[::-1]: node = node.setdefault(ch, {})# reverse char path
            node['$'] = word
        self.stream = deque()
    def query(self, letter: str) -> bool:
        self.stream.appendleft(letter)
        node = self.trie
        for ch in self.stream:
            if '$' in node: return True
            if not ch in node: return False
            node = node[ch]
        return '$' in node

# LC2034. Stock Price Fluctuation
class StockPrice:  # all O(logn)
    def __init__(self):
        self.time2price = SortedDict()
        self.prices = SortedList()
    def update(self, timestamp: int, price: int) -> None:
        if timestamp in self.time2price.keys():  # remove old price and add new price
            self.prices.remove(self.time2price[timestamp])
            self.time2price[timestamp] = price
            self.prices.add(price)
        else:  # add new price
            self.time2price[timestamp] = price
            self.prices.add(price)
    def current(self) -> int:
        return self.time2price[self.time2price.keys()[-1]]
    def maximum(self) -> int: return self.prices[-1]
    def minimum(self) -> int: return self.prices[0]

# LC981. Time Based Key-Value Store
class TimeMap:
    def __init__(self):
        self.M = collections.defaultdict(list)
    def set(self, key: str, value: str, timestamp: int) -> None:
        self.M[key].append((timestamp, value))  # order is same as timestamp, so we could bisect later
    def get(self, key: str, timestamp: int) -> str:
        A = self.M.get(key, None)
        if A is None: return ""
        i = bisect.bisect(A, (timestamp, chr(127)))
        return A[i-1][1] if i else ""

# LC359. Logger Rate Limiter
class Logger:
    def __init__(self):  self.m2t = {}
    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        if message not in self.m2t or timestamp - self.m2t[message] >= 10:
            self.m2t[message] = timestamp
            return True
        else: return False
# LC341. Flatten Nested List Iterator
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.stack = list(reversed(nestedList))  # so pop() O(1)
    def next(self) -> int:
        self.make_stack_top_an_integer()
        return self.stack.pop().getInteger()
    def hasNext(self) -> bool:
        self.make_stack_top_an_integer()
        return len(self.stack) > 0
    def make_stack_top_an_integer(self):
        # While the stack contains a nested list at the top...
        while self.stack and not self.stack[-1].isInteger():
            # Unpack the list at the top by putting its items onto
            # the stack in reverse order.
            self.stack.extend(reversed(self.stack.pop().getList()))


# LC244. Shortest Word Distance II
class WordDistance:
    def __init__(self, words: List[str]):
        self.d = defaultdict(list)
        for i, w in enumerate(words): self.d[w].append(i)
        self.memo = {}
    def shortest(self, word1: str, word2: str) -> int:
        key = (min(word1,word2), max(word1, word2))
        if key in self.memo: return self.memo[key]
        self.memo[key] = min([abs(a - b) for a, b in itertools.product(self.d[word1], self.d[word2])])
        return self.memo[key]


# LC642. Design Search Autocomplete System
from collections import defaultdict
class AutocompleteSystem:
    def __init__(self, sentences: List[str], times: List[int]):
        self.counts = defaultdict(int) # or use trie
        for s, c in zip(sentences, times): self.counts[s] = c
        self.ui = '' # user input
        self.matches = []
    def input(self, c: str) -> List[str]:
        if c == '#':
            self.counts[self.ui] += 1
            self.ui = ''
            self.matches = []
            return []
        if not self.ui: # new input
            self.matches = [(-ct, s) for s, ct in self.counts.items() if s[0] == c]  # O(n)
            self.matches.sort()  # O(nlogn)
            self.matches = [s for _, s in self.matches]
        else:
            n = len(self.ui)
            self.matches = [m for m in self.matches if len(m) > n and m[n] == c]
        self.ui += c
        return self.matches[:3]

# LC1146. Snapshot Array
class SnapshotArray(object):  # This copies only relevant changes
    def __init__(self, length):
        self.dic = defaultdict(dict)
        self.snap_id = 0
    def set(self, index, val):  # O(1)
        self.dic[self.snap_id][index] = val
    def snap(self):
        self.snap_id += 1
        self.dic[self.snap_id] = self.dic[self.snap_id - 1].copy() # copy only set values
        return self.snap_id -1
    def get(self, index, snap_id): # O(1)
        if index in self.dic[snap_id]: return self.dic[snap_id][index]
        else: return 0

# LC706. Design HashMap

# LC284. Peeking Iterator
class PeekingIterator:
    def __init__(self, iterator):
        self.i = iterator
        self.v = iterator.next() if iterator.hasNext() else None
    def peek(self):
        return self.v
    def next(self):
        r = self.v
        self.v = self.i.next() if self.i.hasNext() else None
        return r
    def hasNext(self):
        return self.v is not None

# LC1656. Design an Ordered Stream
class OrderedStream:
    def __init__(self, n: int):
        self.ptr = 1
        self.stream = [None] * (n+2) # extra to align 1-n
    def insert(self, idKey: int, value: str) -> List[str]:
        self.stream[idKey] = value
        if idKey == self.ptr:
            while self.stream[self.ptr] is not None:
                self.ptr += 1 # move forward when value is filled
        return self.stream[idKey:self.ptr]

# LC901. Online Stock Span
class StockSpanner:
    def __init__(self):
        self.peaks = []
    def next(self, price: int) -> int:
        weight = 1 # itself
        while self.peaks and self.peaks[-1][0] <= price:
            weight += self.peaks.pop()[1]
        self.peaks.append((price, weight))
        return weight

# LC1396. Design Underground System
import statistics
from collections import defaultdict
class UndergroundSystem:  # O(1)
    def __init__(self):
        self._checkins = dict()  # book keeping
        self._travels = defaultdict(list)  # stats
    def checkIn(self, id: int, stationName: str, t: int) -> None:
        if id in self._checkins: raise Exception('already checked in')
        self._checkins[id] = (stationName, t)
    def checkOut(self, id: int, stationName: str, t: int) -> None:
        checkin = self._checkins[id]
        del self._checkins[id]
        self._travels[(checkin[0], stationName)].append(t - checkin[1])
    def getAverageTime(self, startStation: str, endStation: str) -> float:
        return statistics.mean(self._travels[(startStation, endStation)])

# LC1472. Design Browser History
class BrowserHistory:
    def __init__(self, homepage: str):
        self.hist = [homepage]
        self.idx = 0
    def visit(self, url: str) -> None:
        self.hist = self.hist[:self.idx+1] # clear forward hist
        self.hist.append(url)
        self.idx = len(self.hist) - 1
    def back(self, steps: int) -> str:
        self.idx -= steps
        if self.idx < 0: self.idx = 0
        return self.hist[self.idx]
    def forward(self, steps: int) -> str:
        self.idx += steps
        if self.idx >= len(self.hist):
            self.idx = len(self.hist) - 1
        return self.hist[self.idx]

