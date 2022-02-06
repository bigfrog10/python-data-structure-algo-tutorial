# LC218. The Skyline Problem
import heapq
def getSkyline(self, buildings):
    events = [(L, -H, R) for L, R, H in buildings]
    events += list({(R, 0, 0) for _, R, _ in buildings})
    events.sort()
    res, hp = [[0, 0]], [(0, float("inf"))]
    for x, negH, R in events:
        while x >= hp[0][1]: heapq.heappop(hp)  # pop up finished buildings
        if negH: heapq.heappush(hp, (negH, R))  # start of a new building
        # record new height: [0][0] is max height
        if res[-1][1] != -hp[0][0]:  res += [x, -hp[0][0]],
    return res[1:]

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

# LC630. Course Schedule III
def scheduleCourse(self, A: List[List[int]]) -> int:
    start, pq = 0, []
    for t, end in sorted(A, key = lambda c: c[1]):
        start += t
        heapq.heappush(pq, -t)
        while start > end:
            start += heapq.heappop(pq)
    return len(pq)
