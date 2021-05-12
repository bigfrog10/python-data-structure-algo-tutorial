# LC380. Insert Delete GetRandom O(1), top100
class RandomizedSet:
    def __init__(self):
        self.values = []
        self.index = {}  # index on positions
    def insert(self, val: int) -> bool:
        if val in self.index: return False
        self.values.append(val)
        self.index[val] = len(self.values) - 1
        return True
    def remove(self, val: int) -> bool:
        if val not in self.index: return False
        pos = self.index[val]
        self.index[val], self.index[self.values[-1]] = self.index[self.values[-1]], self.index[val]
        del self.index[val] # handle index
        self.values[pos], self.values[-1] = self.values[-1], self.values[pos]
        self.values.pop()  # swap to last and remove
        return True
    def getRandom(self) -> int:
        idx = random.randrange(0, len(self.values))  # exclude right end
        return self.values[idx]

# LC381. Insert Delete GetRandom O(1) - Duplicates allowed
class RandomizedCollection: # 93%, fast
    def __init__(self):
        self.values = []
        self.idxs = defaultdict(set) # value to indices
        self.counter = 0
    def insert(self, val: int) -> bool:
        exist = val in self.idxs
        self.values.append(val)
        self.counter += 1
        self.idxs[val].add(self.counter - 1)
        return not exist
    def remove(self, val: int) -> bool:
        if val not in self.idxs: return False
        # we need to be careful when there is only 1 element, this order is important.
        idx = self.idxs[val].pop() # to be removed
        last = self.values[-1] # remove len - 1 for last element
        self.values[idx] = last # move last value to the removed spot
        self.idxs[last].add(idx)
        self.idxs[last].remove(self.counter - 1) # remove last idx, it is len - 1
        # this has to be done after last's index is updated. Otherwise, fail with 1 element.
        if len(self.idxs[val]) == 0: del self.idxs[val]
        self.values.pop()  # remove last element
        self.counter -= 1
        return True
    def getRandom(self) -> int:
        return random.choice(self.values)
