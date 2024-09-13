
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
    if not s or not t or len(s) < len(t): return ""
    counts, included = [0] * 128, len(t)  # counts of each char
    for char in t: counts[ord(char)] += 1  # t counter
    start = end = 0
    start_index, min_len = 0, float('inf')  # compute these 2
    while end < len(s):
        if counts[ord(s[end])] > 0: included -= 1  # only relevant to t
        counts[ord(s[end])] -= 1
        end += 1
        while included == 0:
            if end - start < min_len:
                start_index = start
                min_len = end - start
            if counts[ord(s[start])] == 0: included += 1
            counts[ord(s[start])] += 1
            start += 1
    return "" if min_len == float('inf') else s[start_index:start_index + min_len]

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
