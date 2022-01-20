
# LC680. Valid Palindrome II
def validPalindrome(self, s: str) -> bool:  # O(n)
    n, i = len(s), 0
    while i < n / 2 and s[i] == s[~i]: i += 1
    s = s[i:n - i]
    # remove left or right char
    return s[1:] == s[1:][::-1] or s[:-1] == s[:-1][::-1]



# LC5. Longest Palindromic Substring
def longestPalindrome(self, s): # best solution 96ms, 98%
    if not s: return 0  # O(n^2)
    start, maxLen = 0, 1
    for i in range(len(s)):
        if i-maxLen >= 1 and s[i-maxLen-1:i+1] == s[i-maxLen-1:i+1][::-1]:
            start = i-maxLen-1 # check substring ended to i
            maxLen += 2 # add 1 on both ends
        elif i-maxLen >=0 and s[i-maxLen:i+1] == s[i-maxLen:i+1][::-1]:
            start=i-maxLen
            maxLen += 1 # add on ith char
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



# LC125. Valid Palindrome
def isPalindrome(self, s: str) -> bool: # ignore non alphanumeric, check is or not
    if s is None or s == '': return True
    char_set = 'abcdefghijklmnopqrstuvwxyz0123456789'
    n = len(s)
    start, end = 0, n - 1
    s = s.lower()
    while start < end:
        while s[start] not in char_set and start < len(s)-1: start += 1
        s1 = '' if s[start] not in char_set else s[start]
        while s[end] not in char_set and end > 0: end -= 1
        s2 = '' if s[end] not in char_set else s[end]
        if s1 == s2:
            start += 1
            end -= 1
        else: return False
    return True

# LC266. Palindrome Permutation
def canPermutePalindrome(self, s: str) -> bool:
    counts = Counter(s)
    odd_count = sum(1 for k, v in counts.items() if v % 2 != 0)
    return odd_count < 2

# LC267. Palindrome Permutation II
def generatePalindromes(self, s: str) -> List[str]:
    counter, res = Counter(s), []

    def backtrack(cur=""):
        if not counter: res.append(cur)
        else:
            for c in list(counter.keys()):
                counter[c] -= 2
                if not counter[c]: del counter[c]
                backtrack(c+cur+c)
                counter[c] += 2

    oddCounts = [c for c in counter if counter[c] % 2] # The characters in counter with odd count
    if not len(oddCounts): backtrack()  # if no odd chars, we can simply backtrack
    if len(oddCounts) == 1: # if exactly one odd char, backtrack with oddChar in the middle of string
        oddChar = oddCounts[0]
        counter[oddChar] -= 1
        if not counter[oddChar]: del counter[oddChar]
        backtrack(oddChar)

    return res

# LC647. Palindromic Substrings - return counts of these
def countSubstrings(self, s: str) -> int: # O(n^2)
    def expand(i, j): # O(n) for the call
        cnt = 0
        while i >= 0 and j < len(s) and s[i] == s[j]:
            cnt += 1
            i -= 1
            j += 1
        return cnt
    total = 0
    for i in range(len(s)): # O(n) for the loop
        total += expand(i, i)  # odd expansion from center
        total += expand(i, i+1)  # even expansion from double center
    return total

# LC1216. Valid Palindrome III
def isValidPalindrome(self, s: str, k: int) -> bool: # O(n^2)
    @lru_cache(None)
    def drop(i, j): # how many modifications we do for palindrome
        if i == j: return 0
        elif i == j-1: return 0 if s[i] == s[j] else 1
        if s[i] == s[j]: return drop(i+1, j-1)
        else:
            drops = min(drop(i+1, j), drop(i, j-1))
            return drops + 1
    ret = drop(0, len(s)-1)
    return ret <= k

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

# LC1328. Break a Palindrome
def breakPalindrome(self, palindrome: str) -> str:
    if not palindrome: return ''
    n = len(palindrome)
    if len(palindrome) == 1: return '' # single char is always palindrome.
    i = n-1  # we need to distinguish aaa and bbb
    for idx, ch in enumerate(palindrome):
        if ch != 'a':  # find first non a and replace it with a
            i = idx
            break
    if i == n-1 or i == n // 2:  # aaa -> aab or aba -> abb
        return palindrome[:-1] + 'b'
    return palindrome[:i] + 'a' + palindrome[i+1:]
