
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

# LC680. Valid Palindrome II - deleting at most one character
def validPalindrome(self, s: str) -> bool:  # O(n)
    n, i = len(s), 0
    while i < n / 2 and s[i] == s[~i]: i += 1
    s = s[i:n - i]  # n - i = ~i + 1
    # remove left or right char
    return s[1:] == s[1:][::-1] or s[:-1] == s[:-1][::-1]

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
    ret = drop(0, len(s)-1)
    return ret <= k

# LC336. Palindrome Pairs
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

# LC647. Palindromic Substrings - return counts of these
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

# LC266. Palindrome Permutation - if one
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

# LC131. Palindrome Partitioning
def partition(self, s: str) -> List[List[str]]:  # O(N^2 * 2^N), when all substrings are palindrome, e.g., 'a'*N
    N = len(s)
    @lru_cache(None)
    def recur(i):
        if i >= N: return [[]]
        result = []
        for j in range(i, N):  # O(N)
            tmp = s[i:j + 1]  # partition 1
            if tmp == tmp[::-1]:  # O(N)  # check if it's a palindrome
                for lst in recur(j + 1): result.append([tmp] + lst)  # partition 2
        return result
    return recur(0)

# LC5. Longest Palindromic Substring
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

# LC516. Longest Palindromic Subsequence
def longestPalindromeSubseq(self, s: str) -> int: # O(n^2)
    @lru_cache(None)
    def solve(b, e): # begin and end of the string
        if b > e: return 0
        if b == e: return 1
        if s[b] == s[e]: return 2 + solve(b+1, e-1)
        else: return max(solve(b+1, e), solve(b, e-1))
    # print(solve.cache_info())
    return solve(0, len(s)-1)

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
