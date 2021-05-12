# LC680. Valid Palindrome II
def validPalindrome(self, s: str) -> bool:
    i = 0
    while i < len(s) / 2 and s[i] == s[~i]: i += 1
    s = s[i:len(s) - i]
    return s[1:] == s[1:][::-1] or s[:-1] == s[:-1][::-1] # remove left or remove right char

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

# LC647. Palindromic Substrings
def countSubstrings(self, s: str) -> int: # counts
    def expand(i, j):
        cnt = 0
        while i >= 0 and j < len(s) and s[i] == s[j]:
            cnt += 1
            i -= 1
            j += 1

        return cnt

    total = 0
    for i in range(len(s)):
        total += expand(i, i)  # odd expansion from center
        total += expand(i, i+1)  # even expansion from double center

    return total

# LC1216. Valid Palindrome III
def isValidPalindrome(self, s: str, k: int) -> bool: # recursion
    @lru_cache(None)
    def drop(s, i, j): # how many modifications we do for palindrome
        if i == j: return 0
        elif i == j-1: return 0 if s[i] == s[j] else 1
        if s[i] == s[j]: return drop(s, i+1, j-1)
        else:
            drops = min(drop(s, i+1, j), drop(s, i, j-1))
            return drops + 1
    ret = drop(s, 0, len(s)-1)
    return ret <= k

# LC5. Longest Palindromic Substring
def longestPalindrome(self, s): # best solution 96ms, 98%
    if not s: return 0
    start, maxLen = 0, 1
    for i in range(len(s)):
        if i-maxLen >= 1 and s[i-maxLen-1:i+1] == s[i-maxLen-1:i+1][::-1]:
            start = i-maxLen-1 # check substring ended to i
            maxLen += 2 # add 1 on both ends
        elif i-maxLen >=0 and s[i-maxLen:i+1] == s[i-maxLen:i+1][::-1]:
            start=i-maxLen
            maxLen += 1 # add on ith char
    return s[start:start+maxLen]

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
