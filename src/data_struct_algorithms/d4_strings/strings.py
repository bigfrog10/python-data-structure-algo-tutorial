

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
    n = len(s)
    half = (n+1) // 2
    counts = Counter(s)
    maxk, maxc = None, -1
    for k, c in counts.items(): # only max matters
        if c > maxc: maxk, maxc = k, c
    if maxc > half: return "" # we could have ababa
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
    if not needle: return 0 # '' or None
    if not haystack: return -1 # order matters, needle is not empty
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

# LC844. Backspace String Compare
def backspaceCompare(self, S: str, T: str) -> bool:
    def build(S):
        ans = []
        for c in S:
            if c != '#': ans.append(c)
            elif ans: ans.pop()
        return "".join(ans)
    return build(S) == build(T)

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

# LC71. Simplify Path
def simplifyPath(self, path: str) -> str:
    stack = []
    for folder in path.split('/'):
        if not folder or folder == '.': continue
        elif folder == '..':
            if stack: stack.pop()
        else: stack.append(folder)
    return '/' + '/'.join(stack)

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

# LC616. Add Bold Tag in String
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
    ret = [''] if n % 2 == 0 else ['0', '1', '8'] # start from center
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
