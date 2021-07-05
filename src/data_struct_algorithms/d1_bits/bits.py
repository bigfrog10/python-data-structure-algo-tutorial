# LC868. Binary Gap
def binaryGap(self, n: int) -> int:
    res, last, step = 0, -1, 0
    while n:
        if n & 1: # have 1 in this bit
            if last >= 0: res = max(res, step - last)
            last = step
        n >>= 1
        step += 1
    return res

# LC898. Bitwise ORs of Subarrays
def subarrayBitwiseORs(self, arr: List[int]) -> int:
    # Assume B[i][j] = A[i] | A[i+1] | ... | A[j]
    #Hash set cur stores all wise B[0][i], B[1][i], B[2][i], B[i][i].
    res, cur = set(), set()
    for i in arr:
        cur = {i | j for j in cur} | {i}
        res |= cur
    return len(res)

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


# LC136. Single Number
def singleNumber(self, nums: List[int]) -> int:
    res = 0
    for i in nums: res ^= i
    return res

# LC137. Single Number II
def singleNumber(self, nums: List[int]) -> int:
    seen_once = seen_twice = 0
    for num in nums:
        # first appearance:
        # add num to seen_once
        # don't add to seen_twice because of presence in seen_once

        # second appearance:
        # remove num from seen_once
        # add num to seen_twice

        # third appearance:
        # don't add to seen_once because of presence in seen_twice
        # remove num from seen_twice
        seen_once = ~seen_twice & (seen_once ^ num)
        seen_twice = ~seen_once & (seen_twice ^ num)
    return seen_once

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
