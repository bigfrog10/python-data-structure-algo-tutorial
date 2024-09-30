
# LC745. Prefix and Suffix Search  prefix suffix search
class WordFilter:
    def __init__(self, words: List[str]):  # O(n * L)
        self.dct = {}  # use prefix#suffix as trie paths
        for i in range(len(words)):
            for j in range(len(words[i])):
                for k in reversed(range(len(words[i]))):
                    self.dct[(words[i][:j+1], words[i][k:])] = i
    def f(self, prefix: str, suffix: str) -> int:
        return self.dct.get((prefix, suffix), -1)

# LC146. LRU Cache, top100
class LRUCache:
    def __init__(self, capacity: int): # use ordered dict, language specific
        self.capacity = capacity
        self.values = collections.OrderedDict()  # ordered by first insert
    def get(self, key: int) -> int:  # O(1)
        if key not in self.values: return -1
        self.values.move_to_end(key)  # O(1)  # last is new, first is old
        return self.values[key]
    def put(self, key: int, value: int) -> None:  # O(1)
        if key in self.values: self.values.move_to_end(key) # update
        self.values[key] = value
        if len(self.values) > self.capacity: self.values.popitem(last=False)  # O(1)

# LC460. LFU Cache, BBG
class LFUCache:
    def __init__(self, capacity: int):
        self.capa = capacity
        self.k2freq = {} # key to node
        self.f2kv = defaultdict(dict)  # count to node
        self.minf = 1
    def get(self, key: int) -> int:
        if key not in self.k2freq: return -1
        x = self.k2freq.pop(key) # get freq
        v = self.f2kv[x].pop(key) # get value
        self.k2freq[key] = x + 1 # update freq
        self.f2kv[x+1][key] = v # move to new freq
        if not self.f2kv[x] and self.minf == x:
            self.minf += 1 # if x is only min, bump up
        return v
    def put(self, key: int, value: int) -> None:
        if self.capa <= 0: return
        if key in self.k2freq:
            self.get(key) # update status
            self.f2kv[self.k2freq[key]][key] = value
            return
        if len(self.k2freq) >= self.capa:
            k = next(iter(self.f2kv[self.minf])) # first in dict
            self.f2kv[self.minf].pop(k)
            self.k2freq.pop(k)
        self.k2freq[key] = 1
        self.f2kv[1][key] = value
        self.minf = 1

# LC432. All O`one Data Structure - we need 2 markers, minf and max, string counters
class Node(object):
    def __init__(self, val, prev=None, next=None):
        self.val = val # counter
        self.prev = prev # double linked list
        self.next = next
        self.keys = set() # keys with this freq

class AllOne: # O(1) for all ops
    def __init__(self):
        self.head = Node(0) # min on head, max on tail
        self.tail = Node(0, self.head)
        self.head.next = self.tail
        self.key2node = defaultdict(lambda: self.head)
    def inc(self, key: str) -> None:
        cur = self.key2node[key] # get curr count, key to freq
        cur.keys.discard(key)
        if cur.val + 1 == cur.next.val: new = cur.next
        else:
            new = Node(cur.val + 1, cur, cur.next)
            new.prev.next = new.next.prev = new
        new.keys.add(key)
        self.key2node[key] = new # inc counter
        if not cur.keys and cur.val != 0: # delete curr count node if empty
            cur.prev.next, cur.next.prev = cur.next, cur.prev
    def dec(self, key: str) -> None:
        if not key in self.key2node: return
        cur = self.key2node[key]
        cur.keys.discard(key)
        self.key2node.pop(key)
        if cur.val > 1:
            if cur.val - 1 == cur.prev.val: new = cur.prev
            else:
                new = Node(cur.val - 1, cur, cur.next)
                new.prev.next = new.next.prev = new
            new.keys.add(key)
            self.key2node[key] = new
        if not cur.keys:
            cur.prev.next, cur.next.prev = cur.next, cur.prev # delete
    def getMaxKey(self) -> str:
        if not self.tail.prev.val: return ''
        return next(iter(self.tail.prev.keys))
    def getMinKey(self) -> str:
        if not self.head.next.val: return ''
        return next(iter(self.head.next.keys))
