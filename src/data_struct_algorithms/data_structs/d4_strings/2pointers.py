# LC3. Longest Substring Without Repeating Characters, top100
def lengthOfLongestSubstring(self, s: str) -> int:
    last = [-1] * 128  # used to track last index of every char. treat space for speed.
    st, ans = 0, 0  # substring starting point and result
    for idx, char in enumerate(s):
        if last[ord(char)] != -1:  # showed before, start new
            st = max(st, last[ord(char)] + 1)  # max across different chars
        ans = max(ans, idx - st + 1)
        last[ord(char)] = idx  # update char's index
    return ans

# LC159. Longest Substring with At Most Two Distinct Characters
def lengthOfLongestSubstringTwoDistinct(self, s):  # best
    d = {}
    start = res = 0
    for i, v in enumerate(s):
        if v in d or len(d) < 2:
            res = max(res, i - start + 1)
        else:
            pos = min(d.values())
            del d[s[pos]]
            start = pos + 1
        d[v] = i
    return res

# LC340. Longest Substring with At Most K Distinct Characters
def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
    if k == 0: return 0
    d = {}
    start = res = 0
    for i, v in enumerate(s):
        if v in d or len(d) < k:
            res = max(res, i - start + 1)
        else:
            pos = min(d.values())
            del d[s[pos]]
            start = pos + 1
        d[v] = i
    return res

# LC76. Minimum Window Substring, top100
def minWindow(self, s, t):
    need, missing = collections.Counter(t), len(t) # count downs
    i = 0
    I, J = 0, float('inf')
    for j, c in enumerate(s, 1): # starting index is 1
        missing -= need[c] > 0
        need[c] -= 1
        if missing == 0: # we found a window that has all t chars
            while i < j and need[s[i]] < 0: # catchup until 0
                need[s[i]] += 1
                i += 1
            if  j - i <= J - I: I, J = i, j
    return '' if J == float('inf') else s[I:J]

# LC727. Minimum Window Subsequence
def minWindow(self, S: str, T: str) -> str: # 2 pointers, fast
    res = ""
    start = -1
    while True:  # if we have several substrings that contain T
        first = start + 1 # move to next occurance
        for c in T: # check whether T's chars are in S, if not return "", O(S)
            start = S.find(c, start + 1) # This finds one substrings at a time
            if start == -1: return res
        start = end = start + 1 # now start has T's last char index
        for c in reversed(T):
            start = S.rfind(c, first, start) # T in [first, start]
        if res == "" or len(res) > end - start: res = S[start:end]

# LC30. Substring with Concatenation of All Words
def findSubstring(self, s: str, words: List[str]) -> List[int]:
    if len(words) == 0: return []
    wl = len(words[0])
    wdc = Counter(words)
    ans = []
    for k in range(wl): # sliding window(s)
        subd = Counter() #defaultdict(int) #{}
        left, count = k, 0
        for j in range(k, len(s)-wl+1, wl):
            tword = s[j:j+wl]
            if tword in wdc:  # valid word
                subd[tword] = subd[tword] + 1
                count += 1
                while subd[tword] > wdc[tword]:
                    subd[s[left:left+wl]] -= 1
                    left += wl
                    count -= 1
                if count == len(words): ans.append(left)
            else: # not valid
                subd = Counter() # {}
                left, count = j + wl, 0
    return ans
