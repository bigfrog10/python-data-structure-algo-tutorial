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

# LC179. Largest Number
def largestNumber(self, nums: List[int]) -> str:
    num = [str(x) for x in nums]
    # cmp(x, y) can be replaced by (x > y) - (x < y)
    # in this problem x is (a+b) and y is (b+a)
    cmp = lambda b, a: ((a+b)>(b+a))-((a+b)<(b+a))
    num.sort(key = functools.cmp_to_key(cmp))
    return ''.join(num).lstrip('0') or '0'
