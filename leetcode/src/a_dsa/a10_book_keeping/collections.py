
# LC1773. Count Items Matching a Rule  item match rule  rulekey rulevalue rule repres
def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
    rule = {'type' : 0, 'color' : 1, 'name' : 2}
    return sum(1 for item in items if item[rule[ruleKey]] == ruleValue)

# LC173. Binary Search Tree Iterator   bst iterator
class BSTIterator:  # save space to O(h)
    def __init__(self, root: Optional[TreeNode]):
        self.stack = []  # stack has all elem once, so O(1) amortized
        self._leftmost(root)   # space is O(h)
    def _leftmost(self, node):
        while node:
            self.stack.append(node)
            node = node.left
    def next(self) -> int:  # O(1) amortized
        node = self.stack.pop()
        self._leftmost(node.right)
        return node.val
    def hasNext(self) -> bool:  # O(1)
        return len(self.stack) > 0

# LC1586. Binary Search Tree Iterator II  bst iterator 2
class BSTIterator:
    def __init__(self, root: TreeNode):
        self.last = root
        self.stack, self.arr = [], []
        self.pointer = -1
    def hasNext(self) -> bool:
        return self.stack or self.last or self.pointer < len(self.arr) - 1
    def next(self) -> int:  # O(h)
        self.pointer += 1
        if self.pointer == len(self.arr):
            while self.last: # predecessors
                self.stack.append(self.last)
                self.last = self.last.left
            curr = self.stack.pop()
            self.last = curr.right
            self.arr.append(curr.val)
        return self.arr[self.pointer]
    def hasPrev(self) -> bool:
        return self.pointer > 0
    def prev(self) -> int:
        self.pointer -= 1
        return self.arr[self.pointer]
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):  # O(n) init, O(1) elsewhere. O(n) space
        self.nodes = []
        self.i = -1
        def inorder(root):
            if root is None: return
            inorder(root.left)
            self.nodes.append(root.val)
            inorder(root.right)
        inorder(root)
    def hasNext(self) -> bool:
        return self.i < len(self.nodes) - 1
    def next(self) -> int:
        self.i += 1
        return self.nodes[self.i]
    def hasPrev(self) -> bool:
        return self.i > 0
    def prev(self) -> int:
        self.i -= 1
        return self.nodes[self.i] if self.i >= 0 else self.nodes[0]
# https://leetcode.com/problems/binary-search-tree-iterator-ii/solutions/5060896/python3-beats-100-2024/?envType=company&envId=facebook&favoriteSlug=facebook-three-months


# LC1357. Apply Discount Every n Orders   discount n order
class Cashier:
    def __init__(self, n: int, discount: int, products: List[int], prices: List[int]):
        self.n = n
        self.discount = discount
        self.num_customers = 0
        self.product_prices = {pid: price for pid, price in zip(products, prices)}
    def getBill(self, product: List[int], amount: List[int]) -> float:
        bill = 0
        for pid, amount in zip(product, amount):
            bill += amount * self.product_prices[pid]
        self.num_customers += 1
        if self.num_customers % self.n == 0:
            bill *= (100.0 - self.discount) / 100.0
        return bill

# LC341. Flatten Nested List Iterator, space O(N + L), N # of integers, L # of lists
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):  # O(N+L)
        self.stack = list(reversed(nestedList))  # so pop() O(1)
    def next(self) -> int:  # O(1) amortize
        self.make_stack_top_an_integer()
        return self.stack.pop().getInteger()
    def hasNext(self) -> bool:  # O(1) amortize
        self.make_stack_top_an_integer()
        return len(self.stack) > 0
    def make_stack_top_an_integer(self):  # amortize O(1) - O(1 + total lists/total integers)
        # While the stack contains a nested list at the top...
        while self.stack and not self.stack[-1].isInteger():
            # Unpack the list at the top by putting its items onto
            # the stack in reverse order.
            self.stack.extend(reversed(self.stack.pop().getList()))

# LC362. Design Hit Counter
class HitCounter:
    def __init__(self):
        self.data = []
    def hit(self, timestamp: int) -> None:
        self.data.append(timestamp)
    def getHits(self, timestamp: int) -> int:
        last = timestamp - 300
        idx = bisect.bisect(self.data, last)
        self.data = self.data[idx:]
        return len(self.data)

# LC380. Insert Delete GetRandom O(1), RandomizedSet, top100   random set
import random
class RandomizedSet:
    def __init__(self):
        self.values = []
        self.index = {}  # value 2 position
    def insert(self, val: int) -> bool: # O(1)
        if val in self.index: return False
        self.values.append(val)
        self.index[val] = len(self.values) - 1
        return True
    def remove(self, val: int) -> bool:  # O(1)
        if val not in self.index: return False
        pos = self.index[val]  # index in values
        self.index[self.values[-1]] = pos
        del self.index[val]  # handle index
        self.values[pos] = self.values[-1]
        self.values.pop()  # swap to last and remove
        return True
    def getRandom(self) -> int:  # O(1), just values[random.randrange(len(values))]
        return random.choice(self.values)

# LC642. Design Search Autocomplete System
from collections import defaultdict
class AutocompleteSystem:
    def __init__(self, sentences: List[str], times: List[int]):
        self.counts = defaultdict(int)  # or use trie
        for s, c in zip(sentences, times): self.counts[s] = c
        self.ui = ''  # user input
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

# LC1352. Product of the Last K Numbers
class ProductOfNumbers:
    def __init__(self):
        self.A = [1]
    def add(self, num: int) -> None:
        if num == 0: self.A = [1]
        else: self.A.append(self.A[-1] * num)
    def getProduct(self, k: int) -> int:
        if k >= len(self.A): return 0
        return self.A[-1] // self.A[-k - 1]

# LC706. Design HashMap design
class Bucket:
    def __init__(self):
        self.bucket = []
    def get(self, key):
        for (k, v) in self.bucket:
            if k == key: return v
        return -1
    def update(self, key, value):
        found = False
        for i, kv in enumerate(self.bucket):
            if key == kv[0]:
                self.bucket[i] = (key, value)
                found = True
                break
        if not found:
            self.bucket.append((key, value))
    def remove(self, key):
        for i, kv in enumerate(self.bucket):
            if key == kv[0]: del self.bucket[i]
class MyHashMap:  # O(N / key_space) time and O(M + K), M: key size in map, K: # of buckets
    def __init__(self):
        self.key_space = 2069
        self.hash_table = [Bucket() for i in range(self.key_space)]
    def put(self, key: int, value: int) -> None:
        hash_key = key % self.key_space
        self.hash_table[hash_key].update(key, value)
    def get(self, key: int) -> int:
        hash_key = key % self.key_space
        return self.hash_table[hash_key].get(key)
    def remove(self, key: int) -> None:
        hash_key = key % self.key_space
        self.hash_table[hash_key].remove(key)

# LC1146. Snapshot Array snapshotarray
class SnapshotArray(object):  # This copies only relevant changes
    def __init__(self, length):
        self.dic = defaultdict(dict)
        self.snap_id = 0
    def set(self, index, val):  # O(1)
        self.dic[self.snap_id][index] = val
    def snap(self):
        self.snap_id += 1
        self.dic[self.snap_id] = self.dic[self.snap_id - 1].copy()  # copy only set values
        return self.snap_id -1
    def get(self, index, snap_id): # O(1)
        if index in self.dic[snap_id]: return self.dic[snap_id][index]
        else: return 0
class SnapshotArray:  # min space but extra log(snaps) for get
    def __init__(self, n: int):  # O(n)
        self.A = [[[-1, 0]] for _ in range(n)]
        self.snap_id = 0
    def set(self, index: int, val: int) -> None:  # O(1)
        self.A[index].append([self.snap_id, val])
    def snap(self) -> int:  # O(1)
        self.snap_id += 1
        return self.snap_id - 1
    def get(self, index: int, snap_id: int) -> int:  # O(log(snaps))
        # if modify several times, ids could be same ids, like 1, 2, 2, 2, 3. We want last 2
        i = bisect.bisect(self.A[index], [snap_id + 1]) - 1
        return self.A[index][i][1]

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

# LC211. Design Add and Search Words Data Structure  word data structure  trie dot search dot search data struct
class WordDictionary:  # much faster
    def __init__(self): self.trie = {}
    def addWord(self, word: str) -> None:
        node = self.trie
        for ch in word: node = node.setdefault(ch, {})  # O(len(word))
        node['$'] = True  # mark end of word
    def search(self, word: str) -> bool:  # O(len(word)) if no ., otherwise, O(N * 26^M), M=len(word), N is # of keys
        def find(word, node) -> bool: # recursion on dot
            for i, ch in enumerate(word):
                if ch in node: node = node[ch]  # char found, go down
                else:
                    if ch == '.':  # we need to check all but $
                        for x in node:
                            if x != '$' and find(word[i + 1:], node[x]):
                                return True
                    return False  # if not . or all checks failed
            return '$' in node
        return find(word, self.trie)

# LC2102. Sequentially Ordinal Rank Tracker
from sortedcontainers import SortedList

class SORTracker:  # all O(logn)
    def __init__(self):
        self.sortedList = SortedList()
        self.i = 0
    def add(self, name: str, score: int) -> None:
        self.sortedList.add([-score, name])
    def get(self) -> str:
        ans = self.sortedList[self.i][1]
        self.i += 1
        return ans

# LC855. Exam Room - set apart from each oterh
class ExamRoom:
    def __init__(self, n: int):
        self.N, self.L = n, []
    def seat(self) -> int:  # O(n)
        N, L = self.N, self.L
        if not L: res = 0
        else:
            d, res = L[0], 0
            for a, b in zip(L, L[1:]): # find largest distance between each interval
                if (b - a) // 2 > d:
                    d, res = (b - a) // 2, (b + a) // 2
            if N - 1 - L[-1] > d: res = N - 1
        bisect.insort(L, res)  # O(n)
        return res
    def leave(self, p: int) -> None:  # O(n)
        self.L.remove(p)



# LC676. Implement Magic Dictionary  - one mistake is allowed
class MagicDictionary(object):
    def _candidates(self, word):
        for i in range(len(word)): yield word[:i] + '*' + word[i+1:]  # *ello, h*llo, etc
    def buildDict(self, words):
        self.words = set(words)  # O(n)
        self.near = collections.Counter(cand for word in words
                                        for cand in self._candidates(word))  # O(nk)
    def search(self, word):
        return any(self.near[cand] > 1 or  # case like [hello, hallo] and search for hello
                   self.near[cand] == 1 and word not in self.words
                   for cand in self._candidates(word))  # O(len(word))
class MagicDictionary:
    def __init__(self): self.trie = {}
    def buildDict(self, dictionary: List[str]) -> None:
        for word in dictionary:  # O(mn), m = len(dictionary), n is max word length
            node = self.trie
            for letter in word: node = node.setdefault(letter, {})
            node[None] = None  # word end marker
    def search(self, word: str) -> bool:  # O(26n)
        def find(node, i, mistakeAllowed):
            if i == len(word): return None in node and not mistakeAllowed
            if word[i] not in node:
                return any(find(node[letter], i+1, False) for letter in node if letter) if mistakeAllowed else False
            if mistakeAllowed:
                return find(node[word[i]], i+1, True) or \
                       any(find(node[letter], i+1, False) for letter in node if letter and letter != word[i])
            return find(node[word[i]], i+1, False)
        return find(self.trie, 0, True)

# LC535. Encode and Decode TinyURL
from random import choices  # URLShortener
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
        self.time2price = SortedDict()  # sortedcontainers
        self.prices = SortedList()
    def update(self, timestamp: int, price: int) -> None:
        if timestamp in self.time2price.keys():  #remove old price and add new price
            self.prices.remove(self.time2price[timestamp])
        self.time2price[timestamp] = price
        self.prices.add(price)
    def current(self) -> int:  # O(logn) with any index, latest price
        return self.time2price[self.time2price.peekitem(-1)[0]]
    def maximum(self) -> int: return self.prices[-1]
    def minimum(self) -> int: return self.prices[0]

# LC981. Time Based Key-Value Store
from sortedcontainers import SortedDict
class TimeMap:
    def __init__(self):
        self.data = collections.defaultdict(SortedDict)
    def set(self, key: str, value: str, timestamp: int) -> None:
        self.data[key][timestamp] = value
    def get(self, key: str, timestamp: int) -> str:
        val = self.data[key]
        i = val.bisect_right(timestamp)
        return val[val.keys()[i-1]] if i else ""
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






# LC284. Peeking Iterator, peek iterator
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

# LC251. Flatten 2D Vector
class Vector2D:
    def __init__(self, vec: List[List[int]]):
        self.vecs = vec[::-1]
        self.row = None
    def next(self) -> int:
        self.hasNext()
        return self.row.pop()
    def hasNext(self) -> bool:
        while not self.row and self.vecs:
            self.row = self.vecs.pop()[::-1]
        return self.row is not None and len(self.row) > 0








