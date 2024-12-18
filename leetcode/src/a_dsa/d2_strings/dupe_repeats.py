
# LC424. Longest Repeating Character Replacement  change char change
def characterReplacement(self, s: str, k: int) -> int:
    freqs = Counter()  # O(mn) time, O(m) space: m=len(unique chars), n=len(s)
    start = max_freq = res = 0
    for end in range(len(s)):
        freqs[s[end]] += 1
        max_freq = max(max_freq, freqs[s[end]])
        while max_freq + k < end - start + 1:  # more than we can cover
            freqs[s[start]] -= 1
            start += 1
        res = max(res, end - start + 1)
    return res

# LC1044. Longest Duplicate Substring long dupe string long dupe substring
def longestDupSubstring(self, S):  # O(nlogn) runtime, O(n) space, hard - Rabin-Karp
    A = [ord(c) - ord('a') for c in S]
    mod = 2**63 - 1
    def test(L):  # find duplicated substrings of length L, O(n)
        cur = reduce(lambda x, y: (x * 26 + y) % mod, A[:L])
        seen = {cur}
        p = pow(26, L, mod)  # without hashing, as L -> N/2, this is O(n^2)
        for i in range(L, len(S)):  # slide this window
            cur = (cur * 26 + A[i] - A[i - L] * p) % mod  # rolling hash
            if cur in seen: return i - L + 1  # return start position
            seen.add(cur)
    res, lo, hi = 0, 0, len(S)  # bisect on length L, O(logn)
    while lo < hi:
        mi = (lo + hi + 1) // 2
        pos = test(mi)
        if pos:
            lo = mi
            res = pos
        else: hi = mi - 1
    return S[res:res + lo]

# LC1209. Remove All Adjacent Duplicates in String II - k duplicates, dupe string
def removeDuplicates(self, s, k):  # O(n)
    stack = [['#', 0]]  # 0 for ignoring when joining at the last
    for c in s:
        if stack[-1][0] == c:
            stack[-1][1] += 1
            if stack[-1][1] == k: stack.pop()  # remove this group
        else: stack.append([c, 1])  # char and count
    return ''.join(c * cnt for c, cnt in stack)

# LC316. Remove Duplicate Letters - dupe chars with smallest order
def removeDuplicateLetters(self, s: str) -> str:  # O(n) time and O(1) space
    last_idx = {c: i for i, c in enumerate(s)}
    stack, seen = [], set()  # O(26) = O(1)
    for i, c in enumerate(s): # O(n)
        if c not in seen:  # keep only one inside
            # if stack's char is larger than current and it's not the last
            # we drop this char and wait for the last
            while stack and c < stack[-1] and i < last_idx[stack[-1]]:
                seen.discard(stack.pop()) # increasing stack
            seen.add(c)
            stack.append(c)
    return ''.join(stack)

# LC1047. Remove All Adjacent Duplicates In String - remove duplicates with 2 chars remove adjacent duplicate removal dupe remov
def removeDuplicates(self, S: str) -> str: # O(n)
    output = []
    for ch in S:
        if output and ch == output[-1]: output.pop()
        else: output.append(ch)
    return ''.join(output)

# LC3. Longest Substring Without Repeating Characters, top100
def lengthOfLongestSubstring(self, s: str) -> int:  # O(n) time, O(26) space for sliding window
    chars = Counter()
    res = left = 0
    for right in range(len(s)):
        r = s[right]
        chars[r] += 1  # step 1. modify counter with condition
        while chars[r] > 1:  # step 2. if invalid, make it valid again
            chars[s[left]] -= 1
            left += 1
        res = max(res, right - left + 1)  # step 3. update result
    return res
# https://leetcode.com/problems/minimum-window-substring/

# LC1100. Find k-length substrings with no repeated characters no repeat char no repeating char
def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
    chars = Counter()
    res = left = 0
    for right in range(len(s)):
        r = s[right]
        chars[r] += 1
        while chars[r] > 1:
            chars[s[left]] -= 1
            left += 1
        if right - left + 1 == k:
            res += 1
            chars[s[left]] -= 1
            left += 1
    return res

# LC1081. Smallest Subsequence of Distinct Characters - lexico small subseq
def removeDuplicateLetters(self, s: str) -> str:  # O(n)
    last_idx = {c: i for i, c in enumerate(s)}
    stack, seen = [], set()
    for i, c in enumerate(s):
        if c not in seen:  # keep only one inside
            # if stack's char is larger than current and it's not the last
            # we drop this char and wait for the last
            while stack and c < stack[-1] and i < last_idx[stack[-1]]:
                seen.discard(stack.pop())  # increasing stack
            seen.add(c)
            stack.append(c)
    return ''.join(stack)



# LC459. Repeated Substring Pattern - repeat string repeat substring multiple copies
def repeatedSubstringPattern(self, s: str) -> bool:  # O(n^2)
    idx = (s + s).find(s, 1)
    return len(s) > idx > 0
def repeatedSubstringPattern(self, s: str) -> bool:  # O(n)
    i, j, n = 1, 0, len(s)  # KMP prefix array
    # 1st zero no use.
    dp = [0] * (n+1)  # dp(i) stores the largest index up to i when we have matches
    while i < n:
        if s[i] == s[j]:  # matched
            i += 1
            j += 1
            dp[i] = j
        elif j == 0: i += 1  # to find 1st repeat
        else: j = dp[j]  # mismatch, then roll back j, e.g. "ababcdababcd"
    return dp[n] and dp[n] % (n - dp[n]) == 0

# LC686. Repeated String Match
def repeatedStringMatch(self, a: str, b: str) -> int:
    times = math.ceil(len(b) / len(a))
    if b in a * times: return times
    elif b in a * (times+1): return times + 1
    return -1

# LC395. Longest Substring with At Least K Repeating Characters - repeat k each char k freq long substring k
def longestSubstring(self, s: str, k: int) -> int:  # O(n^2)
    counts = Counter(s)
    for c in set(s):  # O(26)
        if counts[c] < k:
            return max(self.longestSubstring(t, k) for t in s.split(c))
    return len(s)
def longestSubstring(self, s: str, k: int) -> int:
    count = 0  # O(26n) time and O(128) space
    for i in range(1, 27):
        count = max(count, self.count(s, k, i))
    return count
def count(self, s, k, numUniqueTarget):
    start = end = numUnique = numNoLessThanK = count = 0
    chMap = [0]*128
    while end < len(s):
        if chMap[ord(s[end])] == 0: numUnique += 1
        chMap[ord(s[end])] += 1
        if chMap[ord(s[end])] == k: numNoLessThanK += 1
        end += 1
        while numUnique > numUniqueTarget:
            if chMap[ord(s[start])] == k: numNoLessThanK -= 1
            chMap[ord(s[start])] -= 1
            if chMap[ord(s[start])] == 0: numUnique -= 1
            start += 1
        if numUnique == numNoLessThanK: count = max(count, end-start)
    return count
# https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/solutions/719383/python-o-n-sliding-window-solution-based-on-template/

# https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/solutions/719383/python-o-n-sliding-window-solution-based-on-template/
def longestSubstring(self, s: str, k: int) -> int:
    count = 0  # O(26n) time and O(128) space
    for i in range(1, 27):
        count = max(count, self.helper(s, k, i))
    return count
def helper(self, s, k, numUniqueTarget):
    start = end = numUnique = numNoLessThanK = count = 0
    chMap = [0]*128
    while end < len(s):
        if chMap[ord(s[end])] == 0: numUnique += 1
        chMap[ord(s[end])] += 1
        if chMap[ord(s[end])] == k: numNoLessThanK += 1
        end += 1
        while numUnique > numUniqueTarget:
            if chMap[ord(s[start])] == k: numNoLessThanK -= 1
            chMap[ord(s[start])] -= 1
            if chMap[ord(s[start])] == 0: numUnique -= 1
            start += 1
        if numUnique == numNoLessThanK: count = max(count, end-start)
    return count


# LC340. Longest Substring with At Most K Distinct Characters
def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
    seen = Counter()  #  O(len(s)) time and O(k) space
    ans = left = 0
    for right in range(len(s)):
        seen[s[right]] += 1
        while len(seen) > k:
            seen[s[left]] -= 1
            if seen[s[left]] == 0: del seen[s[left]]
            left += 1
        ans = max(ans, right - left + 1)
    return ans



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


























