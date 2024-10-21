
# LC1344. Angle Between Hands of a Clock - clock angles clock
def angleClock(self, hour: int, minutes: int) -> float:
    hour_angle = (hour % 12 + minutes / 60) * 30  # 30 is hour angle 360 / 12
    minutes_angle = minutes * 6  # 6 is minute angle 360 / 60
    diff = abs(hour_angle - minutes_angle)
    return min(diff, 360 - diff)

# LC204. Count Primes  prime number of prime numbers prime numbers of prime
def countPrimes(self, n: int) -> int: # Sieve of Eratosthenes
    import math
    if n <= 2: return 0
    flag = [1] * n  # assume all primes
    flag[0] = flag[1] = 0  # non prime
    upper = math.ceil(math.sqrt(n))
    for i in range(2, upper):
        if flag[i] == 0: continue # non prime
        else:
            m = (n - 1 - i) // i  # every step with i size, how many steps
            flag[2*i::i] = [0] * m
    return sum(flag)

# LC1152. Analyze User Website Visit Pattern user website user pattern
def mostVisitedPattern(self, username: List[str], timestamp: List[int], website: List[str]) -> List[str]:
    # https://leetcode.com/problems/analyze-user-website-visit-pattern/discuss/957611/Python-Solution
    user_visits = defaultdict(list)
    for u, t, w in sorted(zip(username, timestamp, website)):  # timestamps used for sorting
        user_visits[u].append(w)
    count = []  # tuples of 3 elements
    for x in user_visits.values():  # x is list of web links
        count += list(set(itertools.combinations(x, 3)))
    counters = sorted(Counter(count).items(), key=lambda x : (-x[1], x[0]))
    return counters[0][0]  # list of (3-seq, count), so [0] is (3-seq, count), next 0 is 3-seq


# LC386. Lexicographical Numbers  lexico numbers sequence from 1,10,11,...
def lexicalOrder(self, n: int) -> List[int]:
    num, res = 1, []
    for _ in range(n):  # Generate numbers from 1 to n
        res.append(num)
        if num * 10 <= n: num *= 10
        else:
            # Adjust the current number by moving up one digit
            while num % 10 == 9 or num >= n: num //= 10
            num += 1  # Increment the number
    return res

# LC1922. Count Good Numbers
def countGoodNumbers(self, n: int) -> int:
    # even places have 5 options(0, 2, 4, 6, 8)
    # odd places have 4 options(2, 3, 5, or 7)
    MOD = (10 ** 9) + 7
    even_places = (n + 1) // 2
    odd_places = n // 2
    even_combos = pow(5, even_places, MOD)
    odd_combos = pow(4, odd_places, MOD)
    return (even_combos * odd_combos) % MOD

# LC2979. Most Expensive Item That Can Not Be Bought
def mostExpensiveItem(self, primeOne: int, primeTwo: int) -> int:
    return primeOne * primeTwo - primeOne - primeTwo
# https://en.wikipedia.org/wiki/Coin_problem
def mostExpensiveItem(self, primeOne: int, primeTwo: int) -> int:
    @cache
    def dfs(num):
        if num in [primeOne, primeTwo]: return True
        if num < 0: return False
        return dfs(num - primeOne) or dfs(num - primeTwo)
    lcm = primeOne * primeTwo
    for i in range(2, lcm)[::-1]:
        if not dfs(i): return i
    return 1

# LC342. Power of Four
def isPowerOfFour(self, n: int) -> bool:
    return n > 0 and n & (n-1) == 0 and n % 3 == 1 # (n & 0x55555555) != 0

# LC887. Super Egg Drop
def superEggDrop(self, k: int, n: int) -> int:
    drops = 0
    # first zero means with zero eggs we can determine only 0 floors
    floors = [0] * (k+1) # max num of floors that can be checked with i eggs(1-based)
    while floors[k] < n:
        for eggs in range(k, 0, -1):
            # floors[eggs][k-1] is the case where the kth drop is not broken
            # floors[eggs-1][k-1] is the case where the kth drop is broken
            # They are exclusive, so we could add them together
            # k = drops is implicit
            floors[eggs] += 1 + floors[eggs-1]
        drops += 1
        print(floors)
    return drops

# LC3317. Find the Number of Possible Ways for an Event possible ways
mod = 10 ** 9 + 7
@cache
def s(n, k):  # number of ways to assign n performers to k stages
    if n < k: return 0  # each stage at least one performer, stirling number
    if k == 1: return 1
    return (k * s(n - 1, k) + s(n - 1, k - 1)) % mod
class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:  # O(n^2) time, space
        res = 0
        for a in range(1, min(n, x) + 1):  # loop all stages
            # pow(y, a) each event can have 1 to y score
            res += math.perm(x, a) * s(n, a) * pow(y, a, mod)
        return res % mod

# LC1405. Longest Happy String
    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        heapify(heap:=[(-a, 'a'), (-b, 'b'), (-c, 'c')])
        n, ch = heappop(heap)
        ans = []
        while n:
            if n == -1 or heap[0][0] < n:  # n is negative, means -n < heap
                ans.append(ch)  # if the number of this char is not the most
                n += 1
            else:
                ans.append(ch*2)
                n += 2
            n, ch = heapreplace(heap, (n, ch))  # pop first, then insert
        return ''.join(ans)
    # https://leetcode.com/problems/longest-happy-string/solutions/5922435/heap-simple-and-clear-solution-12-lines-runtime-100





