
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
