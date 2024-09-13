import heapq

# LC218. The Skyline Problem
def getSkyline(self, buildings):  # O(nlogn)
    events = [(L, -H, R) for L, R, H in buildings] # -H for heap operations
    events += list({(R, 0, 0) for _, R, _ in buildings})  # to pop finished buildings
    events.sort()  # O(nlogn)
    res, hp = [[0, 0]], [(0, float("inf"))]  # hp: -H, R
    for x, negH, R in events:
        while x >= hp[0][1]: heapq.heappop(hp)  # pop up finished buildings
        if negH: heapq.heappush(hp, (negH, R))  # start of a new building
        # record new height: [0][0] is max height, if there is a change of height
        if res[-1][1] != -hp[0][0]:  res += [x, -hp[0][0]],
    return res[1:]

# LC1453. Maximum Number of Darts Inside of a Circular Dartboard
def numPoints(self, points: List[List[int]], r: int) -> int:  # O(n^2 logn) time
    ans = 1
    for x, y in points:  # O(n^2) for this step, logn below
        angles = []  # O(n) space
        for x1, y1 in points:
            if (x1 != x or y1 != y) and (d:=sqrt((x1-x)**2 + (y1-y)**2)) <= 2*r:
                angle = atan2(y1-y, x1-x)
                delta = acos(d/(2*r))
                angles.append((angle-delta, +1)) #entry
                angles.append((angle+delta, -1)) #exit
        angles.sort(key=lambda x: (x[0], -x[1]))  # O(logn)
        val = 1
        for _, entry in angles:
            val += entry
            ans = max(ans, val)
    return ans
def numPoints(self, A, r):
    res = 1  # O(n^3) time and O(1) space
    for (x1, y1), (x2, y2) in itertools.combinations(A, 2):  # O(n^2)
        d = ((x1 - x2)**2 + (y1 - y2)**2) / 4.0
        if d > r * r: continue
        u = (r * r - d)**0.5  # distance between center and line passing these 2 points
        v = (d * 4) ** 0.5  # distance between 2 points
        x0 = (x1 + x2) / 2.0 + (y2 - y1) * u / v
        y0 = (y1 + y2) / 2.0 - (x2 - x1) * u / v
        # compute only 1 center is enough, because other points will cover this.
        res = max(res, sum((x - x0)**2 + (y - y0)**2 <= r * r + 0.00001 for x, y in A))
    return res

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


