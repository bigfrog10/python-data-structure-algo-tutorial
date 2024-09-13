

# LC893. Groups of Special-Equivalent Strings
def numSpecialEquivGroups(self, words: List[str]) -> int:
    groups = set()
    for word in words:
        groups.add((''.join(sorted(word[1::2])), ''.join(sorted(word[::2]))))
    return len(groups)

# LC422. Valid Word Square
def validWordSquare(self, words: List[str]) -> bool:
    n = len(words)
    for i in range(n):
        for j in range(len(words[i])):  # word in words may not have same length
            try:
                if words[i][j] != words[j][i]: return False
            except: return False
    return True

# LC722. Remove Comments
def removeComments(self, source):
    return filter(None, re.sub('//.*|/\*(.|\n)*?\*/', '', '\n'.join(source)).split('\n'))
def removeComments(self, source: List[str]) -> List[str]:
    in_comment, output = False, []
    for s in source:
        if not in_comment: code = ""
        i = 0
        while i < len(s):
            if i != len(s) - 1 and s[i] + s[i+1] == "//" and not in_comment:
                break # skip comment line
            if i != len(s) - 1 and s[i] + s[i+1] == "/*" and not in_comment:
                in_comment = True
                i += 2
                continue
            if i != len(s) - 1 and s[i] + s[i+1] == "*/" and in_comment:
                in_comment = False
                i += 2
                continue
            if not in_comment:
                code += s[i]
            i += 1
        if code != "" and not in_comment:
            output.append(code)
    return output

# LC474. Ones and Zeroes - knapsack 0/1 1s and 0s and 1s
def findMaxForm(self, strs: List[str], m: int, n: int) -> int:  # O(mnk) time and space, k: len of strs
    xy = [[s.count("0"), s.count("1")] for s in strs]
    @lru_cache(None)
    def dp(zeros, ones, idx):
        if zeros < 0 or ones < 0: return -float("inf")
        if idx == len(strs): return 0
        x, y = xy[idx]
        return max(1 + dp(zeros-x, ones-y, idx + 1), dp(zeros, ones, idx + 1))
    return dp(m, n, 0)

# LC721. Accounts Merge - emails merge  - account merge
def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
    graph = defaultdict(list)  # build graph for emails  n = len(accounts), k=max(len(account))
    for acct in accounts:  ## O(MlogM), M total number of emails M=nk, one account with all emails
        for email in acct[2:]:
            graph[acct[1]].append(email)  # space is O(nk)
            graph[email].append(acct[1])
    def dfs(i):  # to cllect all relevant emails
        tmp = {i}
        for j in graph[i]:
            if j not in seen:
                seen.add(j)
                tmp |= dfs(j)
        return tmp
    seen, ret = set(), []
    for acct in accounts:
        for email in acct[1:]:
            if email not in seen:
                seen.add(email)
                eg = dfs(email)
                ret.append([acct[0]] + sorted(eg))  # O(nlogn), n = (accounts #) * max(email length)
    return ret

# LC621. Task Scheduler - with cooling period
# https://leetcode.com/problems/task-scheduler/solution/
def leastInterval(self, tasks: List[str], n: int) -> int:
    freqs = [0] * 26  # frequencies of the tasks
    for t in tasks: freqs[ord(t) - ord('A')] += 1
    freqs.sort()
    f_max = freqs.pop()  # pop is max
    idle_time = (f_max - 1) * n  # -1 because there is no idle in the last section
    while freqs and idle_time > 0:
        idle_time -= min(f_max - 1, freqs.pop())
    idle_time = max(0, idle_time)
    return idle_time + len(tasks)
def leastInterval(self, tasks: List[str], n: int) -> int:
    frequencies = [0] * 26  # frequencies of the tasks. Only max freq matters
    for t in tasks: frequencies[ord(t) - ord('A')] += 1
    f_max = max(frequencies)  # max frequency, e.g., A, B with max 3, so 3 returned
    n_max = frequencies.count(f_max)  # count of how many tasks with most-frequent, such as A, B
    # at least len(tasks) if no repeat. with repeat we need extra cooling
    # (n+1) tasks in each group with cooling, (f_max-1) groups,
    # -1 is for last group, and it is n_max without need of cooling
    return max(len(tasks), (f_max - 1) * (n + 1) + n_max)

# LC2365. Task Scheduler II
def taskSchedulerII(self, tasks: List[int], space: int) -> int:
    start_day = {task:0 for task in tasks}
    day = 0
    for task in tasks:
        day += 1
        # if the current day is too early to complete the task,
        # fast forward the day to the earliest day you can.
        if day < start_day[task]: day = start_day[task]
        # update the earliest day you can complete the task.
        start_day[task] = day + space + 1
    return day

# LC387. First Unique Character in a String
def firstUniqChar(self, s: str) -> int:
    count = collections.Counter(s)
    for idx, ch in enumerate(s):
        if count[ch] == 1: return idx
    return -1

# LC752. Open the Lock - open lock
def openLock(self, deadends: List[str], target: str) -> int:  # O(n^2 * 10^n + D), N is Number of dials, 4
    def nbs(digit):  # neighbours
        d = int(digit)
        d1 = d - 1 if d > 0 else 9
        d2 = d + 1 if d < 9 else 0
        return str(d1), str(d2)
    terminals = set(deadends)
    queue, seen = deque([('0000', 0)]), {'0000'}  # BFS, num of turns
    while queue:
        state, level = queue.popleft()
        if state == target: return level
        if state in terminals: continue  # deadend, don't go further
        for i, s in enumerate(state):
            for nb in nbs(s):
                nstate = state[:i] + nb + state[i+1:]
                if nstate not in seen:
                    seen.add(nstate)
                    queue.append([nstate, level+1])
    return -1

# LC344. Reverse String
def reverseString(self, s: List[str]) -> None:
    t = len(s)
    for i in range(t // 2): s[i], s[~i] = s[~i], s[i]
    # t = len(s)
    # for i in range(t // 2):
    #     s[i], s[t-1-i] = s[t-1-i], s[i]

# LC205. Isomorphic Strings
def isIsomorphic(self, s: str, t: str) -> bool:
    return len(set(s)) == len(set(zip(s, t))) == len(set(t))

# LC771. Jewels and Stones
def numJewelsInStones(self, jewels: str, stones: str) -> int:
    jset = set(jewels)  # O(n + m)
    return sum(s in jset for s in stones)

# LC1529. Bulb Switcher IV
def minFlips(self, target: str) -> int:
    flips = 0
    status = '0'
    for c in target:
        if c != status:
            flips += 1
            status = '0' if status == '1' else '1' # flip
    return flips

# LC488. Zuma Game
def findMinStep(self, board: str, hand: str) -> int: # O(nm)
    def shrink(s: str): # or use itertools groupby
        lp = 0
        for i in range(len(s)):
            if s[i] != s[lp]:
                if i - lp > 2:
                    return shrink(s[:lp] + s[i:])
                lp = i
        if len(s) - lp > 2: return s[:lp]
        return s
    steps, visited = 0, set()
    que = deque([(board, hand)])
    while que:
        for _ in range(len(que)):
            target, letters = que.popleft()
            if target == '': return steps
            for i, c in enumerate(letters):
                for j in range(len(target)):
                    nt = target[:j] + c + target[j:]
                    if nt in visited: continue
                    visited.add(nt)
                    nt1 = shrink(nt)
                    visited.add(nt1)
                    # if len(nt1) >= len(nt): continue  # 'G', 'GG' -> ''
                    que.append((nt1, letters[:i] + letters[i+1:]))
        steps += 1
    return -1

# LC838. Push Dominoes, 2 pointers
def pushDominoes(self, d): # O(n)
    d = 'L' + d + 'R'
    res = ""
    i = 0
    for j in range(1, len(d)):
        if d[j] == '.': continue
        if i: res += d[i]  # R or L
        middle = j - i - 1
        if d[i] == d[j]: res += d[i] * middle # all same in the middle
        elif d[i] == 'L' and d[j] == 'R': # fall outward
            res += '.' * middle
        else: # fall inward, half L, half R
            res += 'R' * (middle // 2) + '.' * (middle % 2) + 'L' * (middle // 2)
        i = j
    return res

# LC58. Length of Last Word
def lengthOfLastWord(self, s: str) -> int:
    p = len(s) - 1
    while p >= 0 and s[p] == ' ': p -= 1  # trim the trailing spaces
    length = 0  # compute the length of last word
    while p >= 0 and s[p] != ' ':
        p -= 1
        length += 1
    return length

# LC821. Shortest Distance to a Character
def shortestToChar(self, S, C):
    n = len(S)
    res = [0 if c == C else n for c in S]
    for i in range(1, n):
        res[i] = min(res[i], res[i - 1] + 1)
    for i in range(n - 2, -1, -1):
        res[i] = min(res[i], res[i + 1] + 1)
    return res

# LC193. Valid Phone Numbers
import re
pattern = re.compile("^(\([0-9]{3}\) |[0-9]{3}-)[0-9]{3}-[0-9]{4}$")
if re.findall(pattern, "120-345-6789"):
    print("120-345-6789")
