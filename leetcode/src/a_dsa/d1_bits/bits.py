from typing import List
import math

# LC1009. Complement of Base 10 Integer  1's complement
# LC476. Number Complement
def bitwiseComplement(self, N: int) -> int:
    if N == 0: return 1
    todo, bit = N, 1
    while todo:
        N = N ^ bit  # flip current bit
        bit = bit << 1  # prepare for the next run
        todo = todo >> 1
    return N

# LC201. Bitwise AND of Numbers Range - and all nums in [m, n]
def rangeBitwiseAnd(self, m: int, n: int) -> int: # O(1) since n jumps
    while m < n: n = n & (n - 1);  # erase uncommon part: m and n: xxx1000000, xxx0111111
    return n
def rangeBitwiseAnd(self, m: int, n: int) -> int:
    # https://leetcode.com/problems/bitwise-and-of-numbers-range/discuss/593403/Bit-masks-no-loops
    return (m & n & - (2 << math.floor(math.log2(n-m))) if m != n else m)

# LC190. Reverse Bits
def reverseBits(self, n: int) -> int:
    ret = 0
    for shift in range(31, -1, -1):
        ret += (n & 1) << shift
        n >>= 1
    return ret

# LC393. UTF-8 Validation
# Char. number range  |        UTF-8 octet sequence
#       (hexadecimal)    |              (binary)
#    --------------------+---------------------------------------------
#    0000 0000-0000 007F | 0xxxxxxx
#    0000 0080-0000 07FF | 110xxxxx 10xxxxxx
#    0000 0800-0000 FFFF | 1110xxxx 10xxxxxx 10xxxxxx
#    0001 0000-0010 FFFF | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
def validUtf8(self, data: List[int]) -> bool:
    i = 0
    while i < len(data):
        skip, mask = 0, 1 << 7  # 10000000
        while data[i] & mask:  # validate leading bits
            skip += 1
            mask >>= 1
        if skip == 1 or skip > 4: return False  # only 0xxx, 110x, 1110, 11110 are valid

        if any(not x & 1 << 7 or x & 1 << 6 for x in data[i+1:i+skip]):
            return False  # 2nd byte has 10xx
        i += max(1, skip)
    return i == len(data)

# LC477. Total Hamming Distance
def totalHammingDistance(self, nums: List[int]) -> int:  # O(n), better than n^2
    # map takes n to create, zip takes 32 (32 bit)  vertical slices
    zipped = zip(*map('{:032b}'.format, nums))  # result 32 element array
    # every pair of (0, 1) contributes 1 to the distance
    return sum(b.count('0') * b.count('1') for b in zipped)  # constant time


# LC191. Number of 1 Bits, Hamming distance   bit 1 in num 1 bit in num
def hammingWeight(self, n: int) -> int:
    res = 0
    while n != 0:
        res += 1
        n &= (n - 1)  # n & (n-1) erase least significant bit
    return res

# LC260. Single Number III  2 nums show once, rest are twice
def singleNumber(self, nums: List[int]) -> List[int]:
    bitmask = 0  # difference between two numbers (x and y) which were seen only once
    for num in nums: bitmask ^= num
    diff = bitmask & (-bitmask)  # rightmost 1-bit diff between x and y to separate x and y
    x = 0
    for num in nums:
        if num & diff: x ^= num  # bitmask which will contain only x
    return [x, bitmask ^ x]

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

# LC458. Poor Pigs
def poorPigs(self, buckets: int, minutesToDie: int, minutesToTest: int) -> int:
    states = minutesToTest // minutesToDie + 1
    return math.ceil(math.log(buckets) / math.log(states))

    # x pigs could test 2^x buckets
    # states^x >= buckets

# LC461. Hamming Distance
def hammingDistance(self, x: int, y: int) -> int:
    return bin(x^y)[2:].count("1")

# LC89. Gray Code
def grayCode(self, n: int) -> List[int]:
    result = [0]
    for i in range(n):
        prev_size = len(result)
        mask = 1 << i
        for j in range(prev_size)[::-1]:
            result.append(mask + result[j])
    return result