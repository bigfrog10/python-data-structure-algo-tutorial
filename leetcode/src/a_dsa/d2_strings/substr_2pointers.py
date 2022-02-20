
# LC340. Longest Substring with At Most K Distinct Characters
def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
    d = {}  # O(1), k elements
    low, ret = 0, 0
    for i, c in enumerate(s):  # O(n)
        d[c] = i
        if len(d) > k:
            low = min(d.values())  # O(k)
            del d[s[low]]
            low += 1
        ret = max(i - low + 1, ret)
    return ret

# LC727. Minimum Window Subsequence
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

# LC3. Longest Substring Without Repeating Characters, top100
def lengthOfLongestSubstring(self, s: str) -> int:
    last = [-1] * 128  # used to track last index of every char. treat space for speed.
    st, ans = 0, 0  # substring starting point and result
    for idx, char in enumerate(s):
        if last[ord(char)] != -1:  # showed before, start new
            st = max(st, last[ord(char)] + 1)  # max across different chars, abba
        ans = max(ans, idx - st + 1)
        last[ord(char)] = idx  # update char's index
    return ans

# LC76. Minimum Window Substring, top100
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

# LC159. Longest Substring with At Most Two Distinct Characters
def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
    d = {} # O(1) space, only 2 elements
    low, ret = 0, 0
    for i, c in enumerate(s): # O(n)
        d[c] = i
        if len(d) > 2:
            low = min(d.values())
            del d[s[low]]
            low += 1
        ret = max(i - low + 1, ret)
    return ret

# LC424. Longest Repeating Character Replacement
def characterReplacement(self, s: str, k: int) -> int:
    maxf = res = 0
    count = collections.Counter()
    for i in range(len(s)):  # sliding window size maxf + k
        count[s[i]] += 1
        maxf = max(maxf, count[s[i]])
        if res < maxf + k: res += 1  # to deal with 2 separate segments, AABBBAA
        else: count[s[i - res]] -= 1  # we can't replace >k chars, so shrink window
    return res
