
# LC1344. Angle Between Hands of a Clock - clock angles clock
def angleClock(self, hour: int, minutes: int) -> float:
    hour_angle = (hour % 12 + minutes / 60) * 30  # 30 is hour angle 360 / 12
    minutes_angle = minutes * 6  # 6 is minute angle 360 / 60
    diff = abs(hour_angle - minutes_angle)
    return min(diff, 360 - diff)

# LC204. Count Primes
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

# LC1152. Analyze User Website Visit Pattern
def mostVisitedPattern(self, username: List[str], timestamp: List[int], website: List[str]) -> List[str]:
    # https://leetcode.com/problems/analyze-user-website-visit-pattern/discuss/957611/Python-Solution
    user_visits = defaultdict(list)
    for u, t, w in sorted(zip(username, timestamp, website)):
        user_visits[u].append(w)
    count = []  # tuples of 3 elements
    for x in user_visits.values():  # x is list of web links
        count += list(set(itertools.combinations(x, 3)))
    counters = sorted(Counter(count).items(), key=lambda x : (-x[1], x[0]))
    return counters[0][0]  # list of (3-seq, count), so [0] is (3-seq, count), next 0 is 3-seq


# LC386. Lexicographical Numbers
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