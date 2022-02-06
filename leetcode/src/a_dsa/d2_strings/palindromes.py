
# LC125. Valid Palindrome
def isPalindrome(self, s: str) -> bool: # ignore non alphanumeric, check is or not
    i, j = 0, len(s) - 1
    while i < j:
        while i < j and not s[i].isalnum(): i += 1
        while i < j and not s[j].isalnum(): j -= 1
        if s[i].lower() != s[j].lower(): return False
        i += 1
        j -= 1
    return True

# LC680. Valid Palindrome II
def validPalindrome(self, s: str) -> bool:  # O(n)
    n, i = len(s), 0
    while i < n / 2 and s[i] == s[~i]: i += 1
    s = s[i:n - i]
    # remove left or right char
    return s[1:] == s[1:][::-1] or s[:-1] == s[:-1][::-1]

# LC1216. Valid Palindrome III
def isValidPalindrome(self, s: str, k: int) -> bool:  # O(n^2)
    @lru_cache(None)
    def drop(i, j):  # how many modifications we do for palindrome
        if i == j: return 0
        if i == j-1: return 0 if s[i] == s[j] else 1
        if s[i] == s[j]: return drop(i+1, j-1)
        else:
            drops = min(drop(i+1, j), drop(i, j-1))
            return drops + 1
    ret = drop(0, len(s)-1)
    return ret <= k

# LC336. Palindrome Pairs
def palindromePairs(self, words: List[str]) -> List[List[int]]:
    lookup = {w:i for i,w in enumerate(words)}
    res = []
    for i, w in enumerate(words):  # O(nk)
        for j in range(len(w)+1):  # for case like "a", "ba"
            pre, pos = w[:j], w[j:]
            rev_pre, rev_pos = pre[::-1], pos[::-1]
            # pre is palindrome, pos is another. != w is for distinct indices
            if pre == rev_pre and rev_pos != w and rev_pos in lookup:
                res.append([lookup[rev_pos], i])
            # pos is palindrome, j != len(w) is to avoid double count for ""
            if j != len(w) and pos == rev_pos and rev_pre != w and rev_pre in lookup:
                res.append([i, lookup[rev_pre]])
    return res

# LC266. Palindrome Permutation
def canPermutePalindrome(self, s: str) -> bool:
    counts = Counter(s)
    odd_count = sum(1 for k, v in counts.items() if v % 2 != 0)
    return odd_count < 2

# LC647. Palindromic Substrings - return counts of these
def countSubstrings(self, s: str) -> int: # O(n^2)
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

# LC267. Palindrome Permutation II
def generatePalindromes(self, s: str) -> List[str]:
    # https://leetcode.com/problems/palindrome-permutation-ii/discuss/1403853/Python-backtracking%3A-build-string-from-the-middle
    counter, res = Counter(s), []  # O((n/2 + 1)!)
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
    if len(palindrome) == 1: return ''  # single char is always palindrome.
    i = n-1  # we need to distinguish aaa and bbb
    for idx, ch in enumerate(palindrome):
        if ch != 'a':  # find first non a and replace it with a
            i = idx
            break
    if i == n-1 or i == n // 2:  # aaa -> aab or aba -> abb
        return palindrome[:-1] + 'b'
    return palindrome[:i] + 'a' + palindrome[i+1:]

# LC131. Palindrome Partitioning
def partition(self, s: str) -> List[List[str]]:
    if not s: return []
    # O(n^3). dp(i) = partitions of s with length <=i
    dp = {0:[[]], 1:[[s[0]]]}
    for i in range(2, len(s)+1):
        r = []
        for j in range(i):
            t = s[j:i]  # partition 1
            if t == t[::-1]:  # check if it's a palindrome
                for p in dp[j]: r.append(p + [t]) # partition 2
        dp[i] = r
    return dp[len(s)]
def partition(self, s):  # O(2^n) when all substrings are palindrome, aaaa
    N=len(s)
    ans, stack = [], []
    def helper(i):
        if i>=N:
            ans.append(stack[:])
            return
        for j in range(i,N):  # O(n)
            if s[i:j+1]==s[i:j+1][::-1]:
                stack.append(s[i:j+1])
                helper(j+1)
                stack.pop()
    helper(0)
    return ans

# LC5. Longest Palindromic Substring
def longestPalindrome(self, s): # best solution 96ms, 98%
    if not s: return 0  # O(n^2)
    start, maxLen = 0, 1
    for i in range(len(s)):  # 2 pointers
        if i - maxLen >= 1 and s[i - maxLen - 1:i + 1] == s[i-maxLen-1:i+1][::-1]:
            start = i-maxLen-1 # check substring ended to i
            maxLen += 2  # add 1 on both ends
        elif i-maxLen >=0 and s[i-maxLen:i+1] == s[i-maxLen:i+1][::-1]:
            start=i-maxLen
            maxLen += 1  # add on ith char
    return s[start:start+maxLen]

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