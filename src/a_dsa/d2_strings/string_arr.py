
# LC621. Task Scheduler
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
    frequencies = [0] * 26  # frequencies of the tasks
    for t in tasks: frequencies[ord(t) - ord('A')] += 1
    f_max = max(frequencies)  # max frequency
    n_max = frequencies.count(f_max)  # count the most frequent tasks
    # at least len(tasks) if no repeat. with repeat we need extra cooling
    # (n+1) tasks in each group with cooling, (f_max-1) groups, last group is n_max
    return max(len(tasks), (f_max - 1) * (n + 1) + n_max)

# LC443. String Compression
def compress(self, chars: List[str]) -> int:
    st = i = 0
    while i < len(chars):
        while i < len(chars) and chars[i] == chars[st]: i += 1
        if i - st == 1:  # now chars are different
            st = i  # if there is only one char, no change
        else:
            count = str(i - st)
            chars[st + 1: i] = count
            st = st + len(count) + 1
            i = st
    return len(chars)

# LC271. Encode and Decode Strings
class Codec:
    def encode(self, strs: [str]) -> str:  # chunk transfer encoding
        return ''.join('%d:' % len(s) + s for s in strs)
    def decode(self, s: str) -> [str]:
        strs, i = [], 0
        while i < len(s):
            j = s.find(':', i)
            i = j + 1 + int(s[i:j])
            strs.append(s[j+1:i])
        return strs
    # escaping
    def encode(self, strs):
        return ''.join(s.replace('|', '||') + ' | ' for s in strs)
    def decode(self, s):
        return [t.replace('||', '|') for t in s.split(' | ')[:-1]]  # -1 ignores last empty

# LC387. First Unique Character in a String
def firstUniqChar(self, s: str) -> int:
    count = collections.Counter(s)
    for idx, ch in enumerate(s):
        if count[ch] == 1: return idx
    return -1



# LC752. Open the Lock
def openLock(self, deadends: List[str], target: str) -> int:
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

# LC557. Reverse Words in a String III
def reverseWords(self, s: str) -> str:
    words = s.split(' ')
    res = ' '.join(w[::-1] for w in words)
    return res

# LC344. Reverse String
def reverseString(self, s: List[str]) -> None:
    t = len(s)
    for i in range(t // 2):
        s[i], s[t-1-i] = s[t-1-i], s[i]

# LC205. Isomorphic Strings
def isIsomorphic(self, s: str, t: str) -> bool:
    return len(set(s)) == len(set(zip(s, t))) == len(set(t))

# LC771. Jewels and Stones
def numJewelsInStones(self, jewels: str, stones: str) -> int:
    jset = set(jewels)  # O(n + m)
    return sum(s in jset for s in stones)

# LC721. Accounts Merge
def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
    graph = defaultdict(list)  # build graph for emails
    for acct in accounts:  # O(n * m), n = len(accts), m = # of emails per account
        for email in acct[2:]:
            graph[acct[1]].append(email)
            graph[email].append(acct[1])
    seen = set()
    def dfs(i):
        tmp = {i}
        for j in graph[i]:
            if j not in seen:
                seen.add(j)
                tmp |= dfs(j)
        return tmp
    ret = []
    for acct in accounts:
        for email in acct[1:]:
            if email not in seen:
                seen.add(email)
                eg = dfs(email)
                ret.append([acct[0]] + sorted(eg))
    return ret

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
