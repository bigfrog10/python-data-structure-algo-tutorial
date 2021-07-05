
# 621. Task Scheduler
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

# LC443. String Compression
def compress(self, chars: List[str]) -> int:
    idx = 0
    for k, g in groupby(chars):
        chars[idx] = k
        idx += 1
        len_g = len(list(g))
        if len_g == 1: continue
        slen = str(len_g)
        chars[idx:idx + len(slen)] = str(slen)
        idx += len(slen)
    return idx

# LC394. Decode String
def decodeString(self, s: str) -> str:
    stack = []
    curr, k = "", 0
    for char in s:
        if char == "[":
            stack.append((curr, k))
            curr, k = "", 0
        elif char == "]":
            last_string, last_k = stack.pop()
            curr = last_string + last_k * curr
        elif char.isdigit(): k = k * 10 + int(char)
        else: curr += char
    return curr

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


# LC91. Decode Ways, top100
def numDecodings(self, s: str) -> int:  # Best, fast and short
    @lru_cache(maxsize=None)
    def walk(idx):
        if idx == len(s) : return 1
        if s[idx] == '0': return 0
        if idx == len(s) - 1: return 1  # This has to be after above check, case: '0'
        ret = walk(idx + 1)
        if int(s[idx: idx+2]) <= 26: ret += walk(idx + 2)
        return ret
    ret = walk(0)
    return ret

# LC205. Isomorphic Strings
def isIsomorphic(self, s: str, t: str) -> bool:
    return len(set(s)) == len(set(zip(s, t))) == len(set(t))






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
    # trim the trailing spaces
    p = len(s) - 1
    while p >= 0 and s[p] == ' ': p -= 1
    # compute the length of last word
    length = 0
    while p >= 0 and s[p] != ' ':
        p -= 1
        length += 1
    return length

# LC387. First Unique Character in a String
def firstUniqChar(self, s: str) -> int:
    ct = [0] * 26  # char counts
    for c in s: ct[ord(c) - ord('a')] += 1
    for idx, c in enumerate(s):
        if ct[ord(c) - ord('a')] == 1: return idx
    return -1



# LC1108. Defanging an IP Address
def defangIPaddr(self, address: str) -> str:
    return '[.]'.join(address.split('.'))

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
