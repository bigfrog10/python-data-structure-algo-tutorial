
# LC71. Simplify Path
def simplifyPath(self, path: str) -> str:
    stack = []
    for folder in path.split('/'):
        if not folder or folder == '.': continue  # skip this
        elif folder == '..':
            if stack: stack.pop()  # go to parent
        else: stack.append(folder)
    return '/' + '/'.join(stack)

# LC408. Valid Word Abbreviation
def validWordAbbreviation(self, word: str, abbr: str) -> bool:
    wl = len(word)
    i, n = 0, "0" # i is index in word
    for c in abbr:
        if c.isdigit():
            if n == c: return False # don't allow abbr starts with 0
            n += c # accumulate digit, "02"
        else: # letter
            i += int(n)  # add counts for previous letter
            n = '0'  # reset counts to 0
            # counts + current > wl or no match for abbr and word
            if i >= wl or word[i] != c: return False
            i += 1 # move to next char
    return i + int(n) == wl

# LC1047. Remove All Adjacent Duplicates In String
def removeDuplicates(self, S: str) -> str: # O(n)
    output = []
    for ch in S:
        if output and ch == output[-1]: output.pop()
        else: output.append(ch)
    return ''.join(output)

# LC1209. Remove All Adjacent Duplicates in String II
def removeDuplicates(self, s, k):  # O(n)
    stack = [['#', 0]]  # 0 for ignoring when joining at the last
    for c in s:
        if stack[-1][0] == c:
            stack[-1][1] += 1
            if stack[-1][1] == k: stack.pop()  # remove this group
        else: stack.append([c, 1])
    return ''.join(c * cnt for c, cnt in stack)

# LC616. Add Bold Tag in String
def addBoldTag(self, s: str, words: List[str]) -> str:
    status = [False] * len(s)
    for word in words:
        start, last = s.find(word), len(word)
        while start != -1: # this word appears multiple places
            for i in range(start, last+start): status[i] = True
            start = s.find(word, start+1)
    i, final = 0, ""
    while i < len(s):
        if status[i]:
            final += "<b>"
            while i < len(s) and status[i]:
                final += s[i]
                i += 1
            final += "</b>"
        else:
            final += s[i]
            i += 1
    return final

# LC2060. Check if an Original String Exists Given Two Encoded Strings
def possiblyEquals(self, s1: str, s2: str) -> bool:
    def gg(s):  # Return possible length
        ans = {int(s)}
        for i in range(1, len(s)): # split digits among s
            ans |= {x+y for x in gg(s[:i]) for y in gg(s[i:])}
        return ans
    @cache  # make it O(n^3) like
    def fn(i, j, diff):  # DFS  # Return True if s1[i:] matches s2[j:] with given differences
        if i == len(s1) and j == len(s2): return diff == 0
        if i < len(s1) and s1[i].isdigit():
            ii = i
            while ii < len(s1) and s1[ii].isdigit(): ii += 1  # get all digits
            return any(fn(ii, j, diff-x) for x in gg(s1[i:ii]))
        elif j < len(s2) and s2[j].isdigit():
            jj = j
            while jj < len(s2) and s2[jj].isdigit(): jj += 1  # get all digits
            return any(fn(i, jj, diff+x) for x in gg(s2[j:jj]))
        elif diff == 0:  # chars, not digits
            if i < len(s1) and j < len(s2) and s1[i] == s2[j]: return fn(i+1, j+1, 0)
        elif diff > 0:
            if i < len(s1): return fn(i+1, j, diff-1)
        else:
            if j < len(s2): return fn(i, j+1, diff+1)
        return False
    return fn(0, 0, 0)  # diff < 0, means s1 has wild chars, > 0 means s2 has wild chars

# LC681. Next Closest Time
def nextClosestTime(self, time: str) -> str:
    hour, minute = time.split(":")
    # Generate all possible 2 digit values. There are at most 16 sorted values here
    nums = sorted(set(hour + minute))
    two_digit_values = [a+b for a in nums for b in nums]
    i = two_digit_values.index(minute)
    if i + 1 < len(two_digit_values) and two_digit_values[i+1] < "60":
        return hour + ":" + two_digit_values[i+1]
    i = two_digit_values.index(hour)
    if i + 1 < len(two_digit_values) and two_digit_values[i+1] < "24":
        return two_digit_values[i+1] + ":" + two_digit_values[0]
    return two_digit_values[0] + ":" + two_digit_values[0]

# LC2019. The Score of Students Solving Math Expression
def scoreOfStudents(self, s: str, answers: List[int]) -> int:
        @functools.lru_cache(None)
        def dp(i, j):  # all possible result for the substring from s[i] to s[j], O(n^2)
            if i == j: return {int(s[i])}  # base case
            res = {}
            for m in range(i + 1, j, 2):  # O(n) ways to break substrings
                for a in dp(i, m - 1):  # truncate to 1000
                    for b in dp(m + 1, j): # truncate to 1000
                        cur = a * b if s[m] == '*' else a + b
                        if cur <= 1000:  # opt with 0 <= answers[i] <= 1000
                            res[cur] = 2
            return res # truncate to 1000
        res = {**dp(0, len(s) - 1), **{eval(s): 5}}
        return sum(res.get(a, 0) for a in answers)

# LC844. Backspace String Compare
def backspaceCompare(self, S: str, T: str) -> bool:  # O(n+m)
    def build(S):
        ans = []
        for c in S:
            if c != '#': ans.append(c)
            elif ans: ans.pop()
        return "".join(ans)
    return build(S) == build(T)

# LC1055. Shortest Way to Form String
def shortestWay(self, source, target):
    idxs = defaultdict(list)
    for i, c in enumerate(source): idxs[c].append(i)
    result, i = 0, 0  # i: next index of source to check
    for c in target:  # O(nlogm)
        if c not in idxs: return -1  # cannot make target if char not in source
        j = bisect.bisect_left(idxs[c], i)  # index in idxs[c] that is >= i
        if j == len(idxs[c]):           # wrap around to beginning of source
            result += 1
            j = 0
        i = idxs[c][j] + 1              # next index in source
    return result if i == 0 else result + 1     # add 1 for partial source

# LC161. One Edit Distance
def isOneEditDistance(self, s: str, t: str) -> bool:
    if s == t: return False
    l1, l2 = len(s), len(t)
    if l1 > l2: return self.isOneEditDistance(t, s)  # force s no longer than t
    if l2 - l1 > 1: return False
    for i in range(len(s)):
        if s[i] != t[i]:
            if l1 == l2: s = s[:i]+t[i]+s[i+1:]  # replacement
            else: s = s[:i]+t[i]+s[i:]  # insertion
            break
    return s == t or s == t[:-1] # delete, s = "", t = 'a'

# LC468. Validate IP Address
def validIPAddress(self, IP: str) -> str:
    def isIPv4(s):
        try: return str(int(s)) == s and 0 <= int(s) <= 255  # prevent leading 0
        except: return False
    def isIPv6(s):
        try: return len(s) <= 4 and int(s, 16) >= 0 # hex
        except: return False
    segs = IP.split(".")
    if len(segs) == 4 and all(isIPv4(i) for i in segs): return "IPv4"
    segs = IP.split(":")
    if len(segs) == 8 and all(isIPv6(i) for i in segs): return "IPv6"
    return "Neither"

# LC1202. Smallest String With Swaps
def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
    class UF:
        def __init__(self, n): self.p = list(range(n))
        def union(self, x, y): self.p[self.find(x)] = self.find(y)
        def find(self, x):
            if x != self.p[x]: self.p[x] = self.find(self.p[x])
            return self.p[x]
    uf, res, m = UF(len(s)), [], defaultdict(list)
    for x,y in pairs: uf.union(x,y)
    for i in range(len(s)): m[uf.find(i)].append(s[i])
    for comp_id in m.keys(): m[comp_id].sort(reverse=True)
    for i in range(len(s)): res.append(m[uf.find(i)].pop())
    return ''.join(res)

# LC884. Uncommon Words from Two Sentences
def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
    c = collections.Counter((s1 + " " + s2).split())
    return [w for w in c if c[w] == 1]

# LC388. Longest Absolute File Path
def lengthLongestPath(self, input: str) -> int:
    ret, tmp = 0, {-1: 0}
    for line in input.split('\n'):
        depth = line.count('\t')
        # len(line) - depth -> remove tabs
        tmp[depth] = tmp[depth - 1] + len(line) - depth
        if line.count('.'):  # \n replaced by /
            ret = max(ret, tmp[depth] + depth)
    return ret

# LC1044. Longest Duplicate Substring
def longestDupSubstring(self, S):  # O(nlogn) runtime, O(n) space
    A = [ord(c) - ord('a') for c in S]
    mod = 2**63 - 1
    def test(L):  # find duplicated substrings of length L, O(n)
        p = pow(26, L, mod)  # without hashing, as L -> N/2, this is O(n^2)
        cur = reduce(lambda x, y: (x * 26 + y) % mod, A[:L])
        seen = {cur}
        for i in range(L, len(S)):  # slide this window
            cur = (cur * 26 + A[i] - A[i - L] * p) % mod  # rolling hash
            if cur in seen: return i - L + 1  # return start position
            seen.add(cur)
    res, lo, hi = 0, 0, len(S)  # bisect on length L, O(logn)
    while lo < hi:
        mi = (lo + hi + 1) // 2
        pos = test(mi)
        if pos:
            lo = mi
            res = pos
        else: hi = mi - 1
    return S[res:res + lo]

# LC459. Repeated Substring Pattern
def repeatedSubstringPattern(self, s: str) -> bool:
    idx = (s + s).find(s, 1)
    return len(s) > idx > -1

# LC471. Encode String with Shortest Length
@functools.lru_cache(None)
def encode(self, s: str) -> str:  # O(n^4)
    i = (s+s).find(s,1) # O(n^2)
    encoded = str(len(s)//i) + '[' + self.encode(s[:i]) + ']' if i<len(s) else s
    splitEncoded = [self.encode(s[:i]) + self.encode(s[i:]) for i in range(1,len(s))]
    return min(splitEncoded + [encoded], key=len)

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

# LC91. Decode Ways, top100
def numDecodings(self, s: str) -> int:  # Best, fast and short
    @lru_cache(maxsize=None)
    def walk(idx):
        if idx == len(s): return 1
        if s[idx] == '0': return 0
        if idx == len(s) - 1: return 1  # This has to be after above check, case: '0'
        ret = walk(idx + 1)
        if int(s[idx: idx+2]) <= 26: ret += walk(idx + 2)
        return ret
    return walk(0)

# LC1108. Defanging an IP Address
def defangIPaddr(self, address: str) -> str:
    return '[.]'.join(address.split('.'))

# LC833. Find And Replace in String
def findReplaceString(self, S, indexes, sources, targets):
    for i, s, t in sorted(zip(indexes, sources, targets), reverse=True):
        S = S[:i] + t + S[i + len(s):] if S[i:i + len(s)] == s else S
    return S

# LC345. Reverse Vowels of a String
def reverseVowels(self, s):
    s = list(s)  # take the vowel sequence, reverse it
    vows = set('aeiouAEIOU')
    l, r = 0, len(s) - 1
    while l < r:
        while l <= r and s[l] not in vows: l += 1
        while l <= r and s[r] not in vows: r -= 1
        if l > r: break
        s[l], s[r] = s[r], s[l]
        l, r = l + 1, r - 1
    return ''.join(s)

# LC72. Edit Distance
def minDistance(self, word1: str, word2: str) -> int:
    @lru_cache(None)  # O(mn) runtime and space
    def levenshtein(i, j):  # distance of word1[:i] and word2[:j]
        if i == 0: return j  # Need to insert j chars
        if j == 0: return i  # Need to delete i chars
        if word1[i-1] == word2[j-1]: return levenshtein(i-1, j-1)
        # delete or replace
        return min(levenshtein(i-1, j), levenshtein(i, j-1), levenshtein(i-1, j-1)) + 1
    return levenshtein(len(word1), len(word2))

def minDistance(self, word1: str, word2: str) -> int:
    # In the above, recursion relies only previous row, so we could save space
    n, m = len(word1), len(word2)
    if n == 0 or m == 0: return max(n, m)
    # dp(j) is the distance between word1[:0] and word2[:j]
    dp = [j for j in range(m+1)] # all inserts
    for i in range(1, n+1):
        prev = dp[0] # prev = dp[i-1][j-1], save before overwritten
        dp[0] = i # all deletes
        for j in range(1, m+1):
            tmp = dp[j] # save for next round
            if word1[i-1] == word2[j-1]: dp[j] = prev # no edit
            else:
                # try to add 1 char to one of both words
                # this update (make them same),
                # + prev=dp[i-1][j-1], new dp[j-1], and old dp[i-1][j]
                dp[j] = 1 + min(prev, dp[j-1], dp[j])
            prev = tmp
    return dp[-1]

# LC6. ZigZag Conversion
def convert(self, s: str, numRows: int) -> str:
    if numRows == 1: return s
    rows = [''] * numRows
    cur_row, down = 0, -1
    for c in s:
        rows[cur_row] += c
        if cur_row == 0 or cur_row == numRows-1:
            down *= -1
        cur_row += down
    return ''.join([row for row in rows])

# LC767. Reorganize String
def reorganizeString(self, s: str) -> str:
    if not s: return "" # O(n) there is no sort
    n, counts = len(s), Counter(s)
    maxk, maxc = None, -1
    for k, c in counts.items(): # only max matters
        if c > maxc: maxk, maxc = k, c
    if maxc > (n+1) // 2: return "" # we could have ababa
    res = [''] * n
    res[:maxc*2:2] = [maxk] * maxc
    i = maxc*2 if maxc * 2 < n else 1
    for k, c in counts.items():
        if k == maxk: continue
        for j in range(c):
            res[i] = k
            i += 2
            if i >= n: i = 1 # revert back to index 1 to fill odd
    return ''.join(res)

# LC97. Interleaving String
def isInterleave(self, s1: str, s2: str, s3: str) -> bool: # O(nm)
    if len(s3) != len(s1) + len(s2): return False
    @lru_cache(None) # without this, it's O(2^(m+n))
    def match_char(s1,s2,s3):
        if not s1: return s2 == s3
        if not s2: return s1 == s3
        return (s1[0] == s3[0] and match_char(s1[1:],s2,s3[1:])) or \
               (s2[0] == s3[0] and match_char(s1, s2[1:],s3[1:]))
    return match_char(s1,s2,s3)

# LC942. DI String Match
def diStringMatch(self, s: str) -> List[int]:
    lo, hi = 0, len(s)
    ans = []
    for x in s:
        if x == 'I':
            ans.append(lo)
            lo += 1
        else:
            ans.append(hi)
            hi -= 1
    return ans + [lo]

# LC28. Implement strStr()
def strStr(self, haystack: str, needle: str) -> int:
    if not needle: return 0  # '' or None
    if not haystack: return -1  # order matters, needle is not empty
    h_len, n_len = len(haystack), len(needle)
    for i in range(h_len - n_len + 1):
        if needle == haystack[i:i+n_len]: return i
    return -1

# LC1071. Greatest Common Divisor of Strings
def gcdOfStrings(self, str1: str, str2: str) -> str:
    # make sure that str1 and str2 must have `Greatest Common Divisor`
    if str1 + str2 != str2 + str1: return ''
    sz1, sz2 = len(str1), len(str2)
    while sz1 != 0 and sz2 != 0: # GCD
        if sz1 > sz2: sz1 = sz1 % sz2
        else: sz2 = sz2 % sz1
    return str1[:sz1] if sz1 else str2[:sz2]

# LC1247. Minimum Swaps to Make Strings Equal
def minimumSwap(self, s1: str, s2: str) -> int:
    # https://leetcode.com/problems/minimum-swaps-to-make-strings-equal/discuss/419874/Simply-Simple-Python-Solution-with-detailed-explanation
    x_y, y_x = 0, 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            if c1 == 'x': x_y += 1
            else: y_x += 1
    if (x_y + y_x) % 2 == 1: return -1
    # Both x_y and y_x count shall either be even or odd to get the result.
    # x_y + y_x should be even
    res = x_y // 2
    res += y_x // 2
    if x_y % 2 == 1: res += 2
    # If there count is odd i.e. we have "xy" and "yx" situation
    # so we need 2 more swaps to make them equal
    return res

# LC165. Compare Version Numbers
def compareVersion(self, version1: str, version2: str) -> int:
    nums1 = version1.split('.')
    nums2 = version2.split('.')
    n1, n2 = len(nums1), len(nums2)
    for i in range(max(n1, n2)): # compare versions
        i1 = int(nums1[i]) if i < n1 else 0
        i2 = int(nums2[i]) if i < n2 else 0
        if i1 != i2: return 1 if i1 > i2 else -1
    return 0 # the versions are equal

# LC1763. Longest Nice Substring
def longestNiceSubstring(self, s: str) -> str:
    if not s: return "" # boundary condition
    ss = set(s)
    for i, c in enumerate(s):
        if c.swapcase() not in ss:
            s0 = self.longestNiceSubstring(s[:i])
            s1 = self.longestNiceSubstring(s[i+1:])
            return max(s0, s1, key=len)
    return s

# LC763. Partition Labels
def partitionLabels(self, s: str) -> List[int]: # O(n) time and space
    max_idx = {letter: i for i, letter in enumerate(s)}
    ret = []
    start = maxl = 0
    for i, e in enumerate(s):  # O(n) time
        maxl = max(maxl, max_idx[e])
        if maxl == i:
            ret.append(i - start + 1)
            start = i+1
    return ret

# LC758. Bold Words in String
def addBoldTag(self, s, dict):
    status = [False]*len(s)
    for word in dict:
        start, last = s.find(word), len(word)
        while start != -1: # this word appears multiple places
            for i in range(start, last+start): status[i] = True
            start = s.find(word, start+1)
    i, final = 0, ""
    while i < len(s):
        if status[i]:
            final += "<b>"
            while i < len(s) and status[i]:
                final += s[i]
                i += 1
            final += "</b>"
        else:
            final += s[i]
            i += 1
    return final

# LC246. Strobogrammatic Number
def isStrobogrammatic(self, num: str) -> bool:
    # only 0, 1, 6, 8, 9 works. 6 and 9 are paired
    rotates = {'0': '0', '1': '1', '8': '8', '6': '9', '9': '6'}
    left, right = 0, len(num)-1
    while left <= right:
        if num[left] not in rotates or rotates[num[left]] != num[right]:
            return False
        left += 1
        right -= 1
    return True

# LC247. Strobogrammatic Number II
def findStrobogrammatic(self, n: int) -> List[str]:
    # set of 0 1 6 8 9
    ret = [''] if n % 2 == 0 else ['0', '1', '8']  # start from center
    for _ in range(n // 2):
        strobo = []
        for s in ret:
            strobo.append('1' + s + '1')
            strobo.append('8' + s + '8')
            strobo.append('6' + s + '9')
            strobo.append('9' + s + '6')
            if len(s) < n - 2:  # 0 can't be at first position
                strobo.append('0' + s + '0')
        ret = strobo  # next wave spreaded from center
    return ret

# LC395. Longest Substring with At Least K Repeating Characters
def longestSubstring(self, s, k): # O(n)
    counts = Counter(s)
    for c in set(s):
        if counts[c] < k:
            return max(self.longestSubstring(t, k) for t in s.split(c))
    return len(s)

# LC1156. Swap For Longest Repeated Character Substring
def maxRepOpt1(self, S):
    # We get the group's key and length first, e.g. 'aaabaaa' -> [[a , 3], [b, 1], [a, 3]
    A = [[c, len(list(g))] for c, g in itertools.groupby(S)]
    # We also generate a count dict for easy look up e.g. 'aaabaaa' -> {a: 6, b: 1}
    count = collections.Counter(S)
    # only extend 1 more, use min here to avoid the case that there's no extra char to extend
    res = max(min(k + 1, count[c]) for c, k in A)
    # merge 2 groups together
    for i in range(1, len(A) - 1):
        # if both sides have the same char and are separated by only 1 char
        if A[i - 1][0] == A[i + 1][0] and A[i][1] == 1:
            # min here serves the same purpose
            res = max(res, min(A[i - 1][1] + A[i + 1][1] + 1, count[A[i + 1][0]]))
    return res
