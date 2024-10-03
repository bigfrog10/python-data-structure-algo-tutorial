
# LC727. Minimum Window Subsequence - min window s.t. T is a seq of the window of S
def minWindow(self, S: str, T: str) -> str: # 2 pointers, fast
    res, start = "", -1  # O(|S|*|T|)
    while True:  # if we have several substrings that contain T
        first = start + 1  # move to next occurance
        # find a sequence in S to include T
        for c in T:  # check whether T's chars are in S, if not return "", O(S)
            start = S.find(c, start + 1)  # This finds one substrings at a time
            if start == -1: return res
        # improve the sequence by truncate left
        start = end = start + 1  # now start has T's last char index
        for c in reversed(T):
            start = S.rfind(c, first, start)  # T in [first, start]
        if res == "" or len(res) > end - start: res = S[start:end]

# LC76. Minimum Window Substring, min window has all chars in target string min win subs
def minWindow(self, s: str, t: str) -> str:
    n, m = len(s), len(t)
    seen = defaultdict(int)  # faster than count
    for c in t: seen[c] += 1  # how many copies we need
    min_len, start = float('inf'), -1
    left = count = 0
    for r in range(n):
        if seen[s[r]] > 0: count += 1
        seen[s[r]] -= 1
        while count == m:
            if r - left + 1 < min_len:
                min_len = r - left + 1
                start = left
            seen[s[left]] += 1  # reverse of before while
            if seen[s[left]] > 0: count -= 1
            left += 1
    return "" if start == -1 else s[start:start + min_len]
# https://leetcode.com/problems/minimum-window-substring/solutions/5822266/sliding-window-handling-edge-cases-explanation-relevant-test-cases/?envType=company&envId=facebook&favoriteSlug=facebook-three-months

# LC392. Is Subsequence issubseq is subseq
def isSubsequence(self, s: str, t: str) -> bool:  # O(|t|)
    n, m = len(s), len(t)
    sp = 0
    for tp in range(m):
        sp += sp < n and s[sp] == t[tp]
    return sp == n
def isSubsequence(self, s: str, t: str) -> bool:
    it = iter(t)
    return all(c in it for c in s)
