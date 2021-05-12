# LC218. The Skyline Problem
import heapq
def getSkyline(self, buildings):
    events = [(L, -H, R) for L, R, H in buildings]
    events += list({(R, 0, 0) for _, R, _ in buildings})
    events.sort()
    res, hp = [[0, 0]], [(0, float("inf"))]
    for x, negH, R in events:
        while x >= hp[0][1]: heapq.heappop(hp) # pop up finished buildings
        if negH: heapq.heappush(hp, (negH, R)) # start of a new building
        # record new height: [0][0] is max height
        if res[-1][1] != -hp[0][0]:  res += [x, -hp[0][0]],
    return res[1:]

# LC155. Min Stack, top100
class MinStack:
    def __init__(self):
        self.values = []  # list end is stack top
    def push(self, x: int) -> None:
        if self.values:
            t = self.values[-1]
            mv = (x, min(x, t[1])) # maintain current min
        else:  mv = (x, x)
        self.values.append(mv)
    def pop(self) -> None:
        self.values.pop()
    def top(self) -> int:
        t = self.values[-1]
        return t[0]
    def getMin(self) -> int:
        t = self.values[-1]
        return t[1]

# LC716. Max Stack
class MaxStack:
    def __init__(self): self.values = []  # list end is stack top
    def push(self, x: int) -> None:
        if self.values:
            t = self.values[-1]
            mv = (x, max(x, t[1])) # maintain current min
        else:  mv = (x, x)
        self.values.append(mv)
    def pop(self) -> int:
        t = self.values.pop()
        return t[0]
    def top(self) -> int:
        t = self.values[-1]
        return t[0]
    def peekMax(self) -> int:
        t = self.values[-1]
        return t[1]
    def popMax(self) -> int:
        stack = []
        maxv = self.values[-1][1]
        removed = False # remove only first max
        while self.values and self.values[-1][1] == maxv:
            t = self.values.pop() # pop out all entries with this max
            if t[0] == maxv and not removed:  removed = True
            else: stack.append(t)
        oldmax = self.values[-1][1] if self.values else float('-inf')
        while stack:
            v, m = stack.pop()
            oldmax = max(oldmax, v) # add back with new max
            self.values.append((v, oldmax))
        return maxv


# LC1167. Minimum Cost to Connect Sticks
def connectSticks(self, sticks: List[int]) -> int:  # same idea, but much faster
    heapq.heapify(sticks) # O(nlogn)
    r = 0
    while len(sticks) > 1: # O(n)
        a = heapq.heappop(sticks)
        b = heapq.heappop(sticks)
        c = a + b
        heapq.heappush(sticks, c) # O(logn)
        r += c
    return r
