# 621. Task Scheduler
def leastInterval(self, tasks: List[str], n: int) -> int:  # with cooldown n
    frequencies = [0] * 26  # frequencies of the tasks
    for t in tasks: frequencies[ord(t) - ord('A')] += 1
    f_max = max(frequencies)  # max frequency
    n_max = frequencies.count(f_max)  # count the most frequent tasks
    return max(len(tasks), (f_max - 1) * (n + 1) + n_max)
def leastInterval(self, tasks: List[str], n: int) -> int:
    freqs = [0] * 26 # frequencies of the tasks
    for t in tasks: freqs[ord(t) - ord('A')] += 1
    freqs.sort()
    f_max = freqs.pop() # pop is max
    idle_time = (f_max - 1) * n # -1 because there is no idle in the last section
    while freqs and idle_time > 0:
        idle_time -= min(f_max - 1, freqs.pop())
    idle_time = max(0, idle_time)
    return idle_time + len(tasks)

# LC70. Climbing Stairs
def climbStairs(self, n: int) -> int: # it's really fib series
    if n < 3: return n
    n1, n2 = 1, 2
    for i in range(n-2): n1, n2 = n2, n1 + n2
    return n2
from functools import lru_cache
def climbStairs(self, n: int) -> int:
    @lru_cache(None)
    def climb(i, n):
        if i > n: return 0
        if i == n: return 1
        return climb(i+1, n) + climb(i+2, n)
    return climb(0, n)
# LC1152. Analyze User Website Visit Pattern
def mostVisitedPattern(self, username: List[str], timestamp: List[int], website: List[str]) -> List[str]:
    # https://leetcode.com/problems/analyze-user-website-visit-pattern/discuss/957611/Python-Solution
    user_visits = defaultdict(list)
    for u, t, w in sorted(zip(username, timestamp, website)):
        user_visits[u].append(w)
    count = [] # tuples of 3 elements
    for x in user_visits.values(): # x is list of web links
        count += list(set(itertools.combinations(x, 3)))
    counters = sorted(Counter(count).items(), key = lambda x : (-x[1], x[0]))
    return counters[0][0]  # list of (3-seq, count), so [0] is (3-seq, count), next 0 is 3-seq

