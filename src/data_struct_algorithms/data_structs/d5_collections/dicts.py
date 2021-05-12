# LC146. LRU Cache, top100
class LRUCache:
    def __init__(self, capacity: int): # use ordered dict, language specific
        self.capacity = capacity
        self.values = OrderedDict()
    def get(self, key: int) -> int:
        if key not in self.values: return -1
        self.values.move_to_end(key)  # LRU
        return self.values[key]
    def put(self, key: int, value: int) -> None:
        if key in self.values: self.values.move_to_end(key)  # LRU
        self.values[key] = value
        if len(self.values) > self.capacity: self.values.popitem(last=False)  # LRU

# LC460. LFU Cache
class LFUCache:
    def __init__(self, capacity: int):
        self.capa = capacity
        self.k2freq = {} # key to node
        self.f2kv = defaultdict(dict) # count to node
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

# LC432. All O`one Data Structure - we need 2 markers, minf and max
