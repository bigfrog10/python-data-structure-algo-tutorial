
# LC93. Restore IP Addresses - chart in solution is interesting
def restoreIpAddresses(self, s: str) -> List[str]:
    res = []
    def add_dot(segs, start):
        if len(segs) == 4:
            if start == len(s): res.append('.'.join(segs))
        else:  # < 4
            for i in range(start, min(start+3, len(s))):
                if s[start] == '0' and i > start: break
                seg = s[start:i+1]
                if 0 <= int(seg) < 256: add_dot(segs +[seg], i+1)
    add_dot([], 0)
    return res

# LC299. Bulls and Cows
def getHint(self, secret: str, guess: str) -> str:
    sc = Counter(secret)
    bulls = cows = 0
    for idx, ch in enumerate(guess):
        if ch in sc:
            if ch == secret[idx]:
                bulls += 1 # 1122, 1222
                cows -= int(sc[ch] <= 0) # remove if it's excessive
            # corresponding characters don't match
            else: cows += int(sc[ch] > 0)
            sc[ch] -= 1
    return f'{bulls}A{cows}B'

# LC7. Reverse Integer
def reverse(self, x: int) -> int:
    pos = x if x >= 0 else -x
    res = 0
    while pos > 0:
        r = pos % 10
        res = res * 10 + r
        if res > 2**31-1 or res < -2**31: return 0
        pos = pos // 10
    return res if x > 0 else -res

# LC279. Perfect Squares, top100. minimal -> BFS
def numSquares(self, n):
    square_nums = [i * i for i in range(1, int(n**0.5)+1)] # list of square numbers that are less than `n`
    queue, level = {n}, 0
    while queue: # BFS
        level += 1
        next_queue = set()
        for remainder in queue: # construct the queue for the next level
            for square_num in square_nums:
                if remainder == square_num: return level  # find the node!
                elif remainder < square_num: break # overed, no need to go further, cut branches
                else: next_queue.add(remainder - square_num)
        queue = next_queue
    return level

# LC202. Happy Number
def isHappy(self, n: int) -> bool:
    if n == 1: return True
    history = set()
    while n not in history and n != 1:
        history.add(n)
        a = 0
        while n > 0:
            n, q = divmod(n, 10)
            a += q * q
        n = a
    return n == 1

# LC50. Pow(x, n)
def myPow(self, x: float, n: int) -> float:  # minimize mutiplications
    if n < 0:
        n = -n
        x = 1 / x
    ret = 1
    f = x
    while n > 0:
        if n % 2 != 0: ret *= f
        f = f * f
        n = n // 2
    return ret

# LC1067. Digit Count in Range
def digitsCount(self, d: int, low: int, high: int) -> int: # O(logn)
    # https://leetcode.com/problems/digit-count-in-range/discuss/367226/Java-Very-Concise-log10(N)-Solution
    def count_less(num): # number of d appearance in all x <= num
        res, pos = 0, 1 # position 1 from right
        while pos <= num:
            left, right = divmod(num, pos) # so num = left1{X}right
            res += (left - d + 9) // 10 * pos # full count
            res += right + 1 if left % 10 == d else 0 # partial count
            res -= pos if not d else 0 # minus leading zero counts
            pos *= 10 # move to next position on left
        return res
    return count_less(high) - count_less(low-1)
# use 21, 22 with 1 or 0

# LC964. Least Operators to Express Number  Greedy
class Solution:
    @lru_cache(maxsize=None)
    def leastOpsExpressTarget(self, x: int, target: int) -> int:
        # x = 3, t = 2, then either 3/3 + 3/3 or 3 - 3/3
        if x > target: return min(target * 2 - 1, (x - target) * 2)
        if x == target: return 0
        times = 0
        total = x
        while total < target:  # greedy
            total *= x
            times += 1
        subtract = float("inf")
        if total - target < target:
            subtract = self.leastOpsExpressTarget(x, total - target) + times + 1
        add = self.leastOpsExpressTarget(x, target - (total // x)) + times
        return min(add, subtract)

# LC357. Count Numbers with Unique Digits
def countNumbersWithUniqueDigits(self, n: int) -> int:
    if not n: return 1 # n=0
    if n > 10: return 0 # there are only 10 digits.
    # n=1, there are 10 nums. n > 9, first digit has 9 options, can't be 0 on leading.
    res, cnt = 10, 9
    for i in range(n-1):
        cnt *= 9 - i
        res += cnt
    return res



# LC9. Palindrome Number
def isPalindrome(self, x: int) -> bool:
    if x < 0 or (x > 0 and x % 10 == 0): return False
    rev = 0
    while x > rev:
        rev = rev * 10 + x % 10
        x = x // 10
    return x == rev or x == rev // 10



# LC38. Count and Say
def countAndSay(self, n):
    s = '1'
    for _ in range(n - 1):
        s = ''.join(str(len(list(group))) + digit for digit, group in itertools.groupby(s))
        print(s)
    return s
def countAndSay(self, n: int) -> str:
    def say(s: str) -> str:
        res = ''
        tmp = None
        count = 0
        for c in s:
            if tmp: # existing
                if tmp == c: count += 1
                else: # different char, so reset tmp and count
                    res += str(count) + tmp
                    tmp = c
                    count = 1
            else: # begining
                tmp = c
                count += 1
        res += str(count) + tmp
        return res
    t = str(1) # start from 1
    for i in range(0, n-1): t = say(t)
    return t

# LC69. Sqrt(x)
def mySqrt(self, x: int) -> int:
    if x == 0: return 0
    if x < 4: return 1 # to ensure sqrt(x) < x / 2
    left, right = 2, x // 2  # first 2 is sqrt(4)
    while left <= right:
        middle = left + (right - left) // 2
        sqr = middle * middle
        if sqr > x: right = middle - 1  # middle is tested in sqr
        elif sqr < x: left = middle + 1
        else: return middle
    return right  # close to sqrt(x)



# LC343. Integer Break - 2 ** 3 < 3 ** 2, use derivative to know max x = e
def integerBreak(self, n: int) -> int:
    if n == 2 or n == 3: return n - 1
    if n % 3 == 0: return 3**(n//3)
    if n % 3 == 1: return 3**(n//3 - 1)*4
    if n % 3 == 2: return 3**(n//3)*2

# LC1025. Divisor Game # BBG
def divisorGame(self, N: int) -> bool:
    return N & 1 == 0


# LC670. Maximum Swap
def maximumSwap(self, num: int) -> int:
    sn = list(str(num))
    last_idx = {int(v): i for i, v in enumerate(sn)} # last index for the value
    for i, v in enumerate(sn): # loop forward to get the largest
        for d in range(9, int(v), -1): # find largest value above v
            if d in last_idx and last_idx[d] > i:
                sn[i], sn[last_idx[d]] = sn[last_idx[d]], sn[i]
                return int(''.join(sn))
    return num

# LC172. Factorial Trailing Zeroes
def trailingZeroes(self, n: int) -> int:
    zero_count = 0
    while n > 0:
        n //= 5
        zero_count += n
    return zero_count

# LC326. Power of Three
def isPowerOfThree(self, n: int) -> bool:
    return n > 0 and 1162261467 % n == 0

def isPowerOfThree(self, n: int) -> bool:
    if n < 1: return False
    while n % 3 == 0: n //= 3
    return n == 1

# https://leetcode.com/discuss/interview-question/406663/Bloomberg-or-Phone-Screen-or-Min-Steps-to-Generate-Number
# divide by 3 or multiply by 2, end with 1.
def mul_2_div_3(n: int):
    steps = 0
    que, visited = deque([n]), {n}
    while que:
        num = que.popleft()
        if num == 1: return steps
        nn = num * 2
        if nn not in visited:
            visited.add(nn)
            que.append(nn)
        nn = num // 3
        if nn not in visited:
            visited.add(nn)
            que.append(nn)
    return -1

# LC829. Consecutive Numbers Sum
def consecutiveNumbersSum(self, N: int) -> int:
    # a+1 + a+2 + ... + a+k = N => N = ka + k(k+1) / 2
    # => N - k(k+1)/2 = ka => N - k(k+1)/2 is dividable by k so we can find a.
    k = 2
    count = 1 # N itself
    while k < N:
        m = N - k * (k+1) // 2
        if m < 0: break
        if m % k == 0: count += 1
        k += 1
    return count

# LC166. Fraction to Recurring Decimal
def fractionToDecimal(self, numerator, denominator):
    n, remainder = divmod(abs(numerator), abs(denominator))
    sign = '-' if numerator*denominator < 0 else ''
    result = [sign + str(n), '.']
    remainders = {}
    while remainder > 0 and remainder not in remainders:
        remainders[remainder] = len(result)
        n, remainder = divmod(remainder*10, abs(denominator))
        result.append(str(n))
    if remainder in remainders:
        idx = remainders[remainder]
        result.insert(idx, '(')
        result.append(')')
    return ''.join(result).rstrip(".")

def fractionToDecimal(self, n, d): # 32ms, beats 100%
    if n % d == 0: return str(n // d)
    p, q = abs(n), abs(d)
    r = p % q
    s, m = '', {} # s is quotient, m records length pre to repeat
    while r and r not in m:
        m[r] = len(s)
        r, s = r * 10 % q, s + str(r * 10 // q) # long division
    frac = str(p // q) + '.' + (s[:m[r]] + '(' + s[m[r]:] + ')' if r else s)
    return ('' if (n > 0) == (d > 0) else '-') + frac
