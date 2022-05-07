
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

# LC76. Minimum Window Substring, min window has all chars in target string
def minWindow(self, s, t):
    need, missing = collections.Counter(t), len(t)  # count downs
    i = 0
    I, J = 0, float('inf')
    for j, c in enumerate(s, 1):  # starting index is 1, window is s[i:j]
        missing -= need[c] > 0
        need[c] -= 1
        if missing == 0:  # we found a window that has all t chars
            while i < j and need[s[i]] < 0:  # catchup until 0
                need[s[i]] += 1
                i += 1
            if j - i <= J - I: I, J = i, j
    return '' if J == float('inf') else s[I:J]

# LC392. Is Subsequence
def isSubsequence(self, s: str, t: str) -> bool:
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
