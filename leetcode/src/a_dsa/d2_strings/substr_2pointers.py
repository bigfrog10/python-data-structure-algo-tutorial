
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
    for c in t: seen[c] += 1
    min_len, start = float('inf'), -1
    l = count = 0
    for r in range(n):
        if seen[s[r]] > 0: count += 1
        seen[s[r]] -= 1
        while count == m:
            if r - l + 1 < min_len:
                min_len = r - l + 1
                start = l
            seen[s[l]] += 1   # reverse of before while
            if seen[s[l]] > 0: count -= 1
            l += 1
    return "" if start == -1 else s[start:start + min_len]
# https://leetcode.com/problems/minimum-window-substring/solutions/5822266/sliding-window-handling-edge-cases-explanation-relevant-test-cases/?envType=company&envId=facebook&favoriteSlug=facebook-three-months

# LC392. Is Subsequence
def isSubsequence(self, s: str, t: str) -> bool:  # O(|t|)
    LEFT_BOUND, RIGHT_BOUND = len(s), len(t)
    p_left = p_right = 0
    while p_left < LEFT_BOUND and p_right < RIGHT_BOUND:
        # move both pointers or just the right pointer
        if s[p_left] == t[p_right]: p_left += 1
        p_right += 1
    return p_left == LEFT_BOUND
def isSubsequence(self, s: str, t: str) -> bool:
    it = iter(t)
    return all(c in it for c in s)
