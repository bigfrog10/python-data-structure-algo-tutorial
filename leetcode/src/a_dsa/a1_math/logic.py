
# LC319. Bulb Switcher
# bulb is on when switching odd number of times. ith bulb switching on rth round only if r divides i.
# So i has odd number of divisors. But divisors comes in pairs, e.g., 36 has (1, 36), (2, 18), ...
# The only way to have odd divisors is squares, such as (6, 6).
def bulbSwitch(self, n: int) -> int:
    return int(sqrt(n))

# LC277. Find the Celebrity
def findCelebrity(self, n: int) -> int:
    # question: does a know b? if yes, then rule out a, if no, then rule out b
    keep = 0
    for i in range(1, n):
        if knows(keep, i): keep = i
        # else leave keep as is.
    for i in range(n):
        if not knows(i, keep) or (i != keep and knows(keep, i)):
            return -1
    return keep

# LC997. Find the Town Judge
def findJudge(self, n: int, trust: List[List[int]]) -> int:  # O(n) time and space
    count = [0] * (n + 1)
    for i, j in trust:
        count[i] -= 1
        count[j] += 1
    for i in range(1, n + 1):
        if count[i] == n - 1: return i
    return -1

# LC1041. Robot Bounded In Circle  robot circle
def isRobotBounded(self, instructions: str) -> bool:
    # north = 0, east = 1, south = 2, west = 3
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    x = y = 0 # Initial position is in the center
    face = 0 # facing north
    for i in instructions:
        if i == "L": face = (face + 3) % 4
        elif i == "R": face = (face + 1) % 4
        else:
            x += dirs[face][0]
            y += dirs[face][1]
    return (x == 0 and y == 0) or face != 0

# LC1884. Egg Drop With 2 Eggs and N Floors
def twoEggDrop(self, n: int) -> int:
    @lru_cache(None)  # O(eggs * floors^2)
    def dp(floor, eggs):
        if eggs == 1 or floor <= 1: return floor # eggs = 1, you need to try the floors one by one
        res = float('inf')
        for f in range(1, floor + 1): # we do not know at which floor the egg will break
            res = min(res, 1 + max(dp(f - 1, eggs - 1), dp(floor - f, eggs))) # broken or not broken
        return res
    return dp(n, 2)

# LC1269. Number of Ways to Stay in the Same Place After Some Steps
def numWays(self, steps: int, arrLen: int) -> int:
    @cache
    def dfs(steps, pos):
        if pos < 0 or pos > arrLen - 1: return 0
        if steps == 0: return 1 if pos == 0 else 0
        steps -= 1
        return dfs(steps, pos - 1) + dfs(steps, pos) + dfs(steps, pos + 1)
    if steps is None or steps < 0 or not arrLen: return 0
    return dfs(steps, 0) % (10 ** 9 + 7)

# LC858. Mirror Reflection
def mirrorReflection(self, p: int, q: int) -> int:
    # find minimal m, n s.t. m * p = n * q\
    # m is even & n is odd => return 0.
    # m is odd & n is odd => return 1.
    # m is odd & n is even => return 2.
    # if m and n are even, keep reducing them to find min.
    while p % 2 == 0 and q % 2 == 0: p, q = p // 2, q // 2
    return 1 - p % 2 + q % 2
