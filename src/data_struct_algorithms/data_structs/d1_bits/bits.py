# LC191. Number of 1 Bits
def hammingWeight(self, n: int) -> int:
    sum = 0
    while n != 0:
        sum += 1
        n &= (n - 1)  # n & (n-1) erase least significant bit
    return sum
# LC190. Reverse Bits
def reverseBits(self, n: int) -> int:
    ret = 0
    for shift in range(31, -1, -1):
        ret += (n & 1) << shift
        n >>= 1
    return ret
# LC268. Missing Number
def missingNumber(self, nums: List[int]) -> int:
    s = sum(nums)
    n = len(nums)
    t = n * (n + 1) // 2
    return t - s
def missingNumber(self, nums):
    missing = len(nums)
    for i, num in enumerate(nums):
        missing ^= i ^ num
    return missing

# LC136. Single Number
def singleNumber(self, nums: List[int]) -> int:
    res = 0
    for i in nums: res ^= i
    return res

# LC137. Single Number II
def singleNumber(self, nums: List[int]) -> int:
    a = b = 0
    for n in nums:
        b = (b ^ n) & ~a
        a = (a ^ n) & ~b
    return b

# LC260. Single Number III
def singleNumber(self, nums: List[int]) -> List[int]:
    # difference between two numbers (x and y) which were seen only once
    bitmask = 0
    for num in nums: bitmask ^= num
    # rightmost 1-bit diff between x and y to separate x and y
    diff = bitmask & (-bitmask)
    x = 0
    for num in nums:
        if num & diff: x ^= num # bitmask which will contain only x
    return [x, bitmask^x]

# LC957. Prison Cells After N Days
def prisonAfterNDays(self, cells: List[int], N: int) -> List[int]:
    if not cells: return []
    n = len(cells)
    arr = int(''.join(str(b) for b in cells), 2)
    cache = {arr: 0} # state to day
    start = end = 0 # Nones
    for d in range(1, N+1):
        arr = ~ ((arr << 1) ^ (arr >> 1))  # !XOR
        arr &= 0x7e  # set head and tail to zero, 01111110
        if arr in cache:
            start = cache[arr]  # cycle start
            end = d
            break
        else: cache[arr] = d
    p = end - start
    if p > 0:  # found a cycle
        r = (N - start) % p # 0 .. start .. end .. r
        for d in range(r):  # we reuse last arr here, which is end
            arr = ~ (arr << 1) ^ (arr >> 1)
            arr &= 0x7e
    return bin(arr)[2:].zfill(n)  # remove leading 0b

