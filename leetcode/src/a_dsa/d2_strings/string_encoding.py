
# LC91. Decode Ways, top100
def numDecodings(self, s: str) -> int:  # Best, fast and short
    @lru_cache(maxsize=None)
    def walk(idx):
        if idx == len(s): return 1
        if s[idx] == '0': return 0
        if idx == len(s) - 1: return 1  # This has to be after above check, case: '0'
        ret = walk(idx + 1)
        if int(s[idx: idx+2]) <= 26: ret += walk(idx + 2)
        return ret
    return walk(0)

# LC471. Encode String with Shortest Length
@functools.lru_cache(None)
def encode(self, s: str) -> str:  # O(n^4)
    i = (s+s).find(s,1) # O(n^2)
    encoded = str(len(s)//i) + '[' + self.encode(s[:i]) + ']' if i<len(s) else s
    splitEncoded = [self.encode(s[:i]) + self.encode(s[i:]) for i in range(1,len(s))]
    return min(splitEncoded + [encoded], key=len)
def encode(self, s: str) -> str:  # O(n^2)
    @lru_cache(None)
    def mincode(s):  # fastest 100%
        if len(s) < 5: return s
        dp = [0]
        for i in range(1, len(s)): # O(n)
            j = dp[i-1]
            while j > 0 and s[i] != s[j]: j = dp[j-1]
            if s[i] == s[j]: j += 1
            dp += [j]
        smallest = s
        for prefix_len in range(len(s), 0, -1):  # O(n)
            repeat_len = prefix_len - dp[prefix_len-1]
            if repeat_len < prefix_len and prefix_len % repeat_len == 0: # we have a repeat
                prefix_compressed = mincode(s[:repeat_len])
                left_compressed = f'{prefix_len // repeat_len}[{prefix_compressed}]'
            else: left_compressed = s[:prefix_len]
            right_compressed = mincode(s[prefix_len:])
            if len(left_compressed) + len(right_compressed) < len(smallest):
                smallest = left_compressed + right_compressed
        return smallest
    return mincode(s)

# LC394. Decode String
def decodeString(self, s: str) -> str:
    stack = []
    curr, k = "", 0
    for char in s:
        if char == "[":
            stack.append((curr, k))
            curr, k = "", 0
        elif char == "]":
            last_string, last_k = stack.pop()
            curr = last_string + last_k * curr
        elif char.isdigit(): k = k * 10 + int(char)
        else: curr += char
    return curr
