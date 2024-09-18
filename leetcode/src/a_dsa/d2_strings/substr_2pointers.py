
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
    if len(s) < len(t): return ''
    tcount, matchCnt = Counter(t), 0,
    resStart, resLen = 0, len(s) + 1
    left = 0
    for right, ch in enumerate(s):
        if ch in tcount:
            tcount[ch] -= 1
            matchCnt += tcount[ch] == 0
        while matchCnt == len(tcount):
            # we found a smaller window, update result
            curWindowLen = right - left + 1
            if curWindowLen < resLen:
                resStart, resLen = left, curWindowLen
            removeCh = s[left]
            left += 1
            if removeCh in tcount:
                matchCnt -= tcount[removeCh] == 0
                tcount[removeCh] += 1
    return s[resStart:resStart + resLen] if resLen <= len(s) else ''

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
