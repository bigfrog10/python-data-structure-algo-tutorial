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

# LC204. Count Primes
def countPrimes(self, n: int) -> int:
    import math
    if n <= 2: return 0
    if n == 3: return 1
    flag = [1] * n  # assume all primes
    flag[0] = flag[1] = 0  # non prime
    upper = math.ceil(math.sqrt(n))
    for i in range(2, upper+1):
        if flag[i] == 0: continue # non prime
        else:
            m = (n - i - 1) // i  # every step with i size, how may steps
            if 2 * i < n: flag[2*i::i] = [0] * m
    return sum(flag)

# LC217. Contains Duplicate
def containsDuplicate(self, nums: List[int]) -> bool:
    showed = set()
    for i in nums:
        if i in showed: return True
        showed.add(i)
    return False
def containsDuplicate(self, nums):
    return len(set(nums)) < len(nums)

# LC69. Sqrt(x)
def mySqrt(self, x: int) -> int:
    if x == 0: return 0
    if x < 4: return 1 # to ensure sqrt(x) < x / 2
    left, right = 2, x // 2  # first 2 is sqrt(4)
    while left <= right:
        middle = left + (right - left) // 2
        sqr = middle * middle
        if sqr > x: right = middle - 1  # middle is tested in sqr
        elif sqr < x: left = middle + 1
        else: return middle
    return right  # close to sqrt(x)


# LC1041. Robot Bounded In Circle
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
