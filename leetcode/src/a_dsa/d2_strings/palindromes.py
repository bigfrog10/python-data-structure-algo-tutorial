
# LC336. Palindrome Pairs - of a word list
def palindromePairs(self, words: List[str]) -> List[List[int]]:  # O(nk^2)
    lookup = {w:i for i,w in enumerate(words)}
    res = []
    for i, w in enumerate(words):  # O(n)
        for j in range(len(w)+1):  # O(k), for case like "a", "ba"
            pre, pos = w[:j], w[j:]
            rev_pre, rev_pos = pre[::-1], pos[::-1]  # O(k)
            # pre is palindrome, pos is another. != w is for distinct indices
            if pre == rev_pre and rev_pos != w and rev_pos in lookup:
                res.append([lookup[rev_pos], i])
            # pos is palindrome, j != len(w) is to avoid double count for "", such as ["ab", "ba"]
            if j != len(w) and pos == rev_pos and rev_pre != w and rev_pre in lookup:
                res.append([i, lookup[rev_pre]])
    return res

# LC125. Valid Palindrome - ignore non alphanumeric, check is or not
def isPalindrome(self, s: str) -> bool:  # O(n)
    i, j = 0, len(s) - 1  #
    while i < j:
        while i < j and not s[i].isalnum(): i += 1
        while i < j and not s[j].isalnum(): j -= 1
        if s[i].lower() != s[j].lower(): return False
        i += 1
        j -= 1
    return True

# LC5. Longest Palindromic Substring long pali sub  lo ps
# https://leetcode.com/problems/longest-palindromic-substring/solutions/5433321/manacher-s-algorithm-explained-building-off-of-the-expand-around-center-approach/
# https://leetcode.com/problems/longest-palindromic-substring/solutions/4212241/98-55-manacher-s-algorithm/
# https://cp-algorithms.com/string/manacher.html
def longestPalindrome1(self, s): # similar, slower, O(n^2)
    def find_diameter(left, right):  # O(n)
        while 0 <= left and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]
    res = ""
    for i in range(len(s)):  # O(n)
        res = max(find_diameter(i, i), find_diameter(i, i+1), res, key=len)
    return res
def longestPalindrome(self, s: str) -> str:  # O(n) time and space, Manacher's Algorithm
    if len(s) <= 1: return s
    Max_Len, Max_Str = 1, s[0]
    s = '#' + '#'.join(s) + '#' # all palindromes have odd length
    dp = [0 for _ in range(len(s))]  # radius for center i
    center = right = 0
    for i in range(len(s)):
        if i < right:  # cache this info, so while loop below can be skipped.
            dp[i] = min(right-i, dp[2*center-i])  # 2*center-i is mirror of i around center
        while i-dp[i]-1 >= 0 and i+dp[i]+1 < len(s) and s[i-dp[i]-1] == s[i+dp[i]+1]:
            dp[i] += 1  # expand the radius, i+ dp[i] + 1 < n -> 2 loops < O(n)
        if i+dp[i] > right:  # update
            center, right = i, i+dp[i]  # push right further to the right
        if dp[i] > Max_Len:
            Max_Len, Max_Str = dp[i], s[i-dp[i]:i+dp[i]+1].replace('#','')
    return Max_Str

# LC516. Longest Palindromic Subsequence - return length
def longestPalindromeSubseq(self, s: str) -> int: # O(n^2) time and space
    @lru_cache(None)
    def solve(b, e): # begin and end of the string, max len of pali seq
        if b > e: return 0
        if b == e: return 1
        if s[b] == s[e]: return 2 + solve(b+1, e-1)
        else: return max(solve(b+1, e), solve(b, e-1))
    # print(solve.cache_info())
    return solve(0, len(s)-1)
def longestPalindromeSubseq(self, s):  # O(n) space and O(n^2) time
    n = len(s)
    dp = [1] * n
    for j in range(1, len(s)):
        pre = dp[j]
        for i in reversed(range(0, j)):
            tmp = dp[i]
            if s[i] == s[j]: dp[i] = 2 + pre if i + 1 <= j - 1 else 2
            else: dp[i] = max(dp[i + 1], dp[i])
            pre = tmp
    return dp[0]

# LC647. Palindromic Substrings - return counts of these  count pali sub
def countSubstrings(self, s: str) -> int:  # O(n^2)
    def expand(i, j):  # O(n) for the call
        cnt = 0
        while i >= 0 and j < len(s) and s[i] == s[j]:
            cnt += 1
            i -= 1
            j += 1
        return cnt
    total = 0
    for i in range(len(s)):  # O(n) for the loop
        total += expand(i, i)  # odd expansion from center
        total += expand(i, i+1)  # even expansion from double center
    return total

# LC680. Valid Palindrome II - deleting at most one character
def validPalindrome(self, s: str) -> bool:  # O(n) time and O(1) space
    n, i = len(s), 0
    while i < n / 2 and s[i] == s[~i]: i += 1  # ~i = -i-1, bitwise negative
    # check ignoring 1st or last char
    return s[i+1:n-i] == s[i+1:n-i][::-1] or s[i:n-i-1] == s[i:n-i-1][::-1]

# LC131. Palindrome Partitioning
def partition(self, s: str) -> List[List[str]]:  # O(N * 2^N), when all substrings are palindrome, e.g., 'a'*N
    n = len(s)
    @functools.lru_cache(None)
    def dfs(start):
        if start == n: return [[]]
        res = []
        for i in range(start, n):  # O(N)
            cur = s[start:i+1]  # O(N) partition 1
            if cur == cur[::-1]:  # O(N)  # check if it's a palindrome
                res += [[cur] + rest for rest in dfs(i+1)]  # partition 2
        return res
    return dfs(0)

# LC1400. Construct K Palindrome Strings
def canConstruct(self, s: str, k: int) -> bool:
    if k > len(s): return False
    # odd count elems have to <=k
    return sum(i & 1 for i in collections.Counter(s).values()) <= k

# LC564. Find the Closest Palindrome - close to given non palindrome
def nearestPalindromic(self, n: str) -> str:
    l = len(n)
    # with different digits width, it must be either 10...01 or 9...9
    candidates = {str(10 ** l + 1), str(10 ** (l - 1) - 1)}
    # the closest must be in middle digit +1, 0, -1, then flip left to right
    prefix = int(n[:(l + 1)//2])
    for i in map(str, (prefix - 1, prefix, prefix + 1)):
        candidates.add(i + [i, i[:-1]][l & 1][::-1])
    candidates.discard(n)
    return min(candidates, key=lambda x: (abs(int(x) - int(n)), int(x)))



# LC1216. Valid Palindrome III - k-palindrome - remove at most k chars
def isValidPalindrome(self, s: str, k: int) -> bool:  # O(n^2) time and O(n) space
    n = len(s)
    dp = [0] * n  # how many modifications we do for palindrome for j
    for i in range(n-1)[::-1]:
        prev = dp[i]  # dp[i+1][j-1]
        for j in range(i+1, n):
            tmp = dp[j]  # dp[i+1][j]
            if s[i] == s[j]: dp[j] = prev  # no change on modifications
            else: dp[j] = 1 + min(dp[j], dp[j-1])  # dp[i+1][j], dp[i][j-1]
            prev = tmp
    return dp[n-1] <= k
def isValidPalindrome(self, s: str, k: int) -> bool:  # O(n^2) time and space
    @lru_cache(None)
    def drop(i, j):  # how many modifications we do for palindrome
        if i == j: return 0
        if i == j-1: return 0 if s[i] == s[j] else 1
        if s[i] == s[j]: return drop(i+1, j-1)
        else: return min(drop(i+1, j), drop(i, j-1)) + 1
    return drop(0, len(s)-1) <= k

# LC266. Palindrome Permutation - if any permuatation can be a palindrome
def canPermutePalindrome(self, s: str) -> bool:  # O(n) runtime, O(1) space
    counts = Counter(s)
    odd_count = sum(1 for k, v in counts.items() if v % 2 != 0)
    return odd_count < 2



# LC267. Palindrome Permutation II - return all such permutations
def generatePalindromes(self, s: str) -> List[str]:
    # https://leetcode.com/problems/palindrome-permutation-ii/discuss/1403853/Python-backtracking%3A-build-string-from-the-middle
    counter, res = Counter(s), []  # O((n/2)!)
    def backtrack(cur):
        if not counter: res.append(cur)
        else:
            for c in list(counter.keys()):
                counter[c] -= 2
                if not counter[c]: del counter[c]
                backtrack(c + cur + c)
                counter[c] += 2
    oddCounts = [c for c in counter if counter[c] % 2]  # The characters in counter with odd count
    if not len(oddCounts): backtrack("")  # if no odd chars, we can simply backtrack
    if len(oddCounts) == 1:  # if exactly one odd char, backtrack with oddChar in the middle of string
        oddChar = oddCounts[0]
        counter[oddChar] -= 1
        if not counter[oddChar]: del counter[oddChar]
        backtrack(oddChar)
    return res

# LC1328. Break a Palindrome
def breakPalindrome(self, palindrome: str) -> str:
    if not palindrome: return ''
    n = len(palindrome)
    if n == 1: return ''  # single char is always palindrome.
    i = n-1  # we need to distinguish aaa and bbb
    for idx, ch in enumerate(palindrome):
        if ch != 'a':  # find first non a and replace it with a
            i = idx
            break
    if i == n-1 or i == n // 2:  # aaa -> aab or aba -> abb
        return palindrome[:-1] + 'b'
    return palindrome[:i] + 'a' + palindrome[i+1:]







# LC1312. Minimum Insertion Steps to Make a String Palindrome
from functools import lru_cache
def minInsertions(self, s: str) -> int:
    @lru_cache(None)
    def dp(left, right): # O(n^2)
        if left >= right: return 0
        if s[left] == s[right]:
            return dp(left+1, right-1)
        else:
            return 1 + min(dp(left+1, right), dp(left, right-1))
    return dp(0, len(s)-1)

# LC730. Count Different Palindromic Subsequences
def countPalindromicSubsequences(self, S: str) -> int:  # O(n^3)
    MOD = 1000000007
    @lru_cache(maxsize=None)
    def compute(start: int, end: int) -> int:  # O(n^2)
        if start >= end: return 0
        count = 0
        for ch in "abcd":
            left, right = S.find(ch, start, end), S.rfind(ch, start, end)  # O(n)
            if left == -1 or right == -1:
                continue
            count += 1 if left == right else 2 + compute(left + 1, right)  # 2 for 'a', 'aa'. recursion for a*a
        return count % MOD
    return compute(0, len(S))
