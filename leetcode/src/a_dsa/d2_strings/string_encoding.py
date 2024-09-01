
# LC443. String Compression
def compress(self, chars: List[str]) -> int:  # O(n) time and O(1) space, chars gets shrinked
    st = i = 0
    while i < len(chars):
        while i < len(chars) and chars[i] == chars[st]: i += 1
        if i - st == 1:  st = i # single diff char, leave it alone
        else:
            count = str(i - st)
            chars[st + 1 : i] = count  # after char then add count
            i = st = st + 1 + len(count)  # skip spaces used by count
    return len(chars)

# LC394. Decode String - expand copies
def decodeString(self, s: str) -> str:  # O(n*k) time, k is max coeff; O(|s| - parenth)
    stack = []
    curr, k = "", 0
    for char in s:  # O(n)
        if char == "[":
            stack.append((curr, k))
            curr, k = "", 0
        elif char == "]":
            last_string, last_k = stack.pop()
            curr = last_string + last_k * curr  # O(k)
        elif char.isdigit(): k = k * 10 + int(char)
        else: curr += char
    return curr

# LC91. Decode Ways, decode 12 to 1,2 and 12
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

# LC2060. Check if an Original String Exists Given Two Encoded Strings
def possiblyEquals(self, s1: str, s2: str) -> bool:
    def comb(s):  # Return possible length
        ans = {int(s)}
        for i in range(1, len(s)): # split digits among s
            ans |= {x+y for x in comb(s[:i]) for y in comb(s[i:])}
        return ans
    @cache  # make it O(n^4) like
    def fn(i, j, diff):  # DFS  # Return True if s1[i:] matches s2[j:] with given differences
        if i == len(s1) and j == len(s2): return diff == 0
        if i < len(s1) and s1[i].isdigit():
            ii = i
            while ii < len(s1) and s1[ii].isdigit(): ii += 1  # get all digits
            return any(fn(ii, j, diff-x) for x in comb(s1[i:ii]))
        elif j < len(s2) and s2[j].isdigit():
            jj = j
            while jj < len(s2) and s2[jj].isdigit(): jj += 1  # get all digits
            return any(fn(i, jj, diff+x) for x in comb(s2[j:jj]))
        elif diff == 0:  # chars, not digits
            if i < len(s1) and j < len(s2) and s1[i] == s2[j]: return fn(i+1, j+1, 0)
        elif diff > 0:
            if i < len(s1): return fn(i+1, j, diff-1)
        else:
            if j < len(s2): return fn(i, j+1, diff+1)
        return False
    return fn(0, 0, 0)  # diff < 0, means s1 has wild chars, > 0 means s2 has wild chars



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



# LC271. Encode and Decode Strings
class Codec:
    def encode(self, strs: [str]) -> str:  # chunk transfer encoding
        return ''.join('%d:' % len(s) + s for s in strs)
    def decode(self, s: str) -> [str]:
        strs, i = [], 0
        while i < len(s):
            j = s.find(':', i)
            i = j + 1 + int(s[i:j])
            strs.append(s[j+1:i])
        return strs
    # escaping
    def encode(self, strs):
        return ''.join(s.replace('|', '||') + ' | ' for s in strs)
    def decode(self, s):
        return [t.replace('||', '|') for t in s.split(' | ')[:-1]]  # -1 ignores last empty

