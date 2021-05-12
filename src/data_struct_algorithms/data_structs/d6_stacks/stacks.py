

# LC844. Backspace String Compare
def backspaceCompare(self, S: str, T: str) -> bool:
    def build(S):
        ans = []
        for c in S:
            if c != '#': ans.append(c)
            elif ans: ans.pop()
        return "".join(ans)
    return build(S) == build(T)

# LC503. Next Greater Element II
def nextGreaterElements(self, nums):  # best solution
    n = len(nums)
    ret = [-1] * n
    stack = nums[::-1] # reverse this because append() below, append is O(1)
    for i in range(n - 1, -1, -1):
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        if stack: ret[i] = stack[-1] # this -1 number is larger than current
        stack.append(nums[i]) # in case this is the next greater
    return ret

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

# LC471. Encode String with Shortest Length
@functools.lru_cache(None)
def encode(self, s: str) -> str:  # O(n^4)
    i = (s+s).find(s,1) # O(n^2)
    encoded = str(len(s)//i) + '[' + self.encode(s[:i]) + ']' if i<len(s) else s

    splitEncoded = [self.encode(s[:i]) + self.encode(s[i:]) for i in range(1,len(s))]
    return min(splitEncoded + [encoded], key=len)

# LC895. Maximum Frequency Stack
class FreqStack(object): # This is O(1) operations
    def __init__(self):
        self.maxf = 0 # This is the trick to keep O(1)
        self.v2f = collections.defaultdict(int)
        self.f2v = collections.defaultdict(list)
    def push(self, x):
        # we don't delete x from current freq, this maintains the stack order.
        self.v2f[x] += 1
        self.f2v[self.v2f[x]].append(x)
        self.maxf = max(self.maxf, self.v2f[x])
    def pop(self):
        x = self.f2v[self.maxf].pop()
        #if that's the only max element in frequency decrease by 1
        if not self.f2v[self.maxf]:
            self.maxf -= 1 # this is true because of the line below, i.e., freq dec by 1
        # we don't need to reinsert x with lower frequency, it's already there.
        self.v2f[x] -= 1
        return x
