
# LC400. Nth Digit
def findNthDigit(self, n: int) -> int:  # O(logn) time since we go by digits
    n -= 1  # index alignment
    for digit in range(1, 11):  # loop groups 10-99, 100-999, ...
        first = 10**(digit - 1)  # the first element in the groups, 1, 10, 100, 1000
        # 9 * first - the size of the group. 9, 90, 900, 9000
        k = 9 * first * digit  # total number of digits in this group, 1＊9, 2＊90, 3＊900, 4＊9000
        # first + n/digits is the number contains nth digit - "digits" is the width of each num in this group
        if n < k: return int(str(first + n // digit)[n % digit])  # first + .. is the number where the digit is
        n -= k

# LC233. Number of Digit One  # https://leetcode.com/problems/number-of-digit-one/
# https://leetcode.com/submissions/detail/725602255/
def countDigitOne(self, n: int) -> int:  # O(logn), see leetcode submission
    res = 0
    k = 1
    while k <= n:
        q, r = divmod(n, k)  # for kth digit from right
        full = (q + 8) // 10 * k
        partial = r + 1 if q % 10 == 1 else 0  # 0 because they are count as full above
        res += full + partial
        k *= 10
    return res

# LC1689. Partitioning Into Minimum Number Of Deci-Binary Numbers
def minPartitions(self, n: str) -> int:  # max digit
    return int(max(n))

# LC1291. Sequential Digits - continuous digits
def sequentialDigits(self, low: int, high: int) -> List[int]:  # O(len(high) - len(low))
    n, sample = 10, "123456789"
    nums = []
    for length in range(len(str(low)), len(str(high)) + 1):
        for start in range(n - length):
            num = int(sample[start: start + length])
            if num >= low and num <= high:
                nums.append(num)
    return nums

# LC231. Power of Two
def isPowerOfTwo(self, n: int) -> bool:
    return n != 0 and n & (n-1) == 0

# LC279. Perfect Squares - min squares sum to n
def numSquares(self, n):
    square_nums = [i * i for i in range(1, int(n**0.5)+1)]  # O(sqrt(n)) space and runtime
    queue, level = {n}, 0
    while queue:  # BFS
        level += 1
        next_queue = set()
        for remainder in queue:  # construct the queue for the next level
            for square_num in square_nums:
                if remainder == square_num: return level  # find the node!
                elif remainder < square_num: break  # overed, no need to go further, cut branches
                else: next_queue.add(remainder - square_num)
        queue = next_queue
    return level

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

# LC93. Restore IP Addresses - chart in solution is interesting
def restoreIpAddresses(self, s: str) -> List[str]:
    res = []  # 3^3 possibilities, start or ., have only 3 possibility to put the next dot.
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

# LC50. Pow(x, n)
def myPow(self, x: float, n: int) -> float:  # O(logn)
    if n < 0: n, x = -n, 1 / x
    ret = 1
    while n > 0:
        if n % 2 != 0: ret *= x
        x, n = x * x, n // 2
    return ret

# LC69. Sqrt(x)
def mySqrt(self, x: int) -> int:
    if x == 0: return 0
    if x < 4: return 1  # to ensure sqrt(x) < x / 2
    left, right = 2, x // 2  # first 2 is sqrt(4)
    while left <= right:
        middle = left + (right - left) // 2
        sqr = middle * middle
        if sqr > x: right = middle - 1  # middle is tested in sqr
        elif sqr < x: left = middle + 1
        else: return middle
    return right  # close to sqrt(x)
def mySqrt(self, x: int) -> int:  # This is a new pattern
    left, right = 0, x
    while left < right:
        mid = (left + right + 1) // 2
        if mid * mid > x: right = mid - 1
        else: left = mid
    return left

# LC670. Maximum Swap - swap digits in number to get max - swap max - max swap
def maximumSwap(self, num: int) -> int:  # O(n)
    sn = list(str(num))
    last_idx = {int(v): i for i, v in enumerate(sn)}  # last index for the value
    for i, v in enumerate(sn):  # loop forward to get the largest
        for d in range(9, int(v), -1):  # find largest value above v
            if d in last_idx and last_idx[d] > i:
                sn[i], sn[last_idx[d]] = sn[last_idx[d]], sn[i]
                return int(''.join(sn))
    return num

# LC2081. Sum of k-Mirror Numbers, k mirror
def kMirror(self, k: int, n: int) -> int:
    @cache   ## why generator does not work
    def generate(start, n):  # n is number of digits, O(k^(n/2))
        if n == 1: return [str(i) for i in range(start, k)]
        elif n == 2: return [str(i) + str(i) for i in range(start, k)]
        else:
            arr = []
            for elem in [str(i) for i in range(start, k)]:
                for center in generate(0, n - 2):
                    arr.append(elem + center + elem)
            return arr
    ans, counter, digits = 0, 0, 1
    while True:
        for num in generate(1, digits):  # start from 1, need positive int
            numBase = str(int(num, k))
            if numBase == numBase[::-1]:
                ans += int(numBase)
                counter += 1
                if counter == n: return ans
        digits += 1

# LC313. Super Ugly Number
def nthSuperUglyNumber(self, n, primes):
    uglies = [1]
    merged = heapq.merge(*map(lambda p: (u*p for u in uglies), primes))
    # merged = list(merged)
    uniqed = (u for u, _ in itertools.groupby(merged))
    any(map(uglies.append, itertools.islice(uniqed, n-1)))
    return uglies[-1]
def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
    res, heap = [1], [(primes[i], primes[i], 0) for i in range(len(primes))]
    while len(res) < n:
        val, prm, idx = heappop(heap)
        if val <= res[-1]:
            while val <= res[-1]: idx += 1; val = prm * res[idx]
        else:
            res += val,
            val, idx = prm * res[idx + 1], idx + 1
        heappush(heap, (val, prm, idx))
    return res[-1]

# LC1492. The kth Factor of n
def kthFactor(self, n: int, k: int) -> int:
    isqrt = math.isqrt(n)
    for i in range(1, isqrt + 1):
        if n % i == 0:
            k -= 1
            if k == 0: return i
    for i in reversed(range(1, isqrt + 1)):
        if i * i == n: continue
        if n % i == 0:
            k -= 1
            if k == 0: return n // i
    return -1

# LC1842. Next Palindrome Using Same Digits
def nextPalindrome(self, num: str) -> str:
    n = len(num)
    k, r = divmod(n, 2)
    mid = num[k] if r else ''

    s = num[:k]  # take 1st half
    stack = []
    for j in range(k-1, -1, -1):  # going backward
        if not stack or s[j] >= s[j+1]: stack += s[j]  # add larger to stack
        else:  # when see smaller element, swap with least larger element
            index = bisect.bisect_right(stack, s[j])
            x, stack[index] = stack[index], s[j]
            sub = s[:j] + x + ''.join(stack)
            return sub + mid + sub[::-1]
    return ''

# LC166. Fraction to Recurring Decimal
def fractionToDecimal(self, n, d): # 32ms, beats 100%
    if n % d == 0: return str(n // d)  # 5/6
    p, q = abs(n), abs(d)  # p 5, q 6
    r = p % q  # 5
    s, m = '', {}  # s is quotient, m records length pre to repeat
    while r and r not in m:
        m[r] = len(s)
        r, s = r * 10 % q, s + str(r * 10 // q)  # long division
    # s = 83, m = 5->0, 2->1, frac = 0.8(3), r = 2
    frac = str(p // q) + '.' + (s[:m[r]] + '(' + s[m[r]:] + ')' if r else s)
    return ('' if (n > 0) == (d > 0) else '-') + frac

# LC556. Next Greater Element III - almost same as next permutation
def nextGreaterElement(self, n: int) -> int:
    digits = list(str(n))
    i = len(digits) - 1
    while i-1 >= 0 and digits[i] <= digits[i-1]: i -= 1  # find first downward
    if i == 0: return -1  # if no downward, return -1
    j = i  # next find smallest larger
    while j+1 < len(digits) and digits[j+1] > digits[i-1]: j += 1
    digits[i-1], digits[j] = digits[j], digits[i-1]  # swap
    digits[i:] = digits[i:][::-1]  # reverse
    ret = int(''.join(digits))
    return ret if ret < 1 << 31 else -1

# LC788. Rotated Digits - num of good integers, strobo
def rotatedDigits(self, N: int) -> int:  # O(logn)
    s1, s2 = {0, 1, 8}, {0, 1, 8, 2, 5, 6, 9}
    res, s = 0, set()
    N = list(map(int, str(N)))  # 157 -> [1, 5, 7]
    for i, v in enumerate(N):  # go through each digit, 1, 5, 7
        for j in range(v):  # for 0, or 1 and then next digit, 0, 1, 2, 3, 4, ...
            if j in s2 and s.issubset(s2): # there are n-i-1 digits remain
                res += 7**(len(N) - i - 1)  # combinations of picking from s2
            if j in s1 and s.issubset(s1):
                res -= 3**(len(N) - i - 1)  # combinations of s1, same number, so discount
        if v not in s2: return res  # if it's 3, 4, 7, then we can't do it after this number
        s.add(v)
    return res + (s.issubset(s2) and not s.issubset(s1))  # last digit



# LC405. Convert a Number to Hexadecimal
def toHex(self, num: int) -> str:
    mappings = "0123456789abcdef"
    result = []
    for _ in range(8):  # 32 bit integer
        num, r = divmod(num, 16)
        result.append(mappings[r])
        if num == 0: break
    return ''.join(result[::-1])





# LC367. Valid Perfect Square
def isPerfectSquare(self, num: int) -> bool:
    if num == 1: return True
    left, right = 2, num // 2
    while left <= right:
        x = left + (right - left) // 2
        guess_squared = x * x
        if guess_squared == num: return True
        if guess_squared > num: right = x - 1
        else: left = x + 1
    return False
def isPerfectSquare(self, num: int) -> bool:
    if num == 1: return True  # Newton's method
    x = num // 2
    while x * x > num:
        x = (x + num // x) // 2  # x_next = x - f(x) / f'(x)
    return x * x == num



# LC246. Strobogrammatic Number - if it is such a number
def isStrobogrammatic(self, num: str) -> bool:
    # only 0, 1, 6, 8, 9 works. 6 and 9 are paired
    rotates = {'0': '0', '1': '1', '8': '8', '6': '9', '9': '6'}
    left, right = 0, len(num)-1
    while left <= right:
        if num[left] not in rotates or rotates[num[left]] != num[right]:
            return False
        left += 1
        right -= 1
    return True

# LC247. Strobogrammatic Number II - return all results
def findStrobogrammatic(self, n: int) -> List[str]:  # O(5^(n/2) * n)
    # set of 0 1 6 8 9
    ret = [''] if n % 2 == 0 else ['0', '1', '8']  # start from center
    for _ in range(n // 2):
        strobo = []
        for s in ret:
            strobo.append('1' + s + '1')  # string append O(n)
            strobo.append('8' + s + '8')
            strobo.append('6' + s + '9')
            strobo.append('9' + s + '6')
            if len(s) < n - 2:  # 0 can't be at first position
                strobo.append('0' + s + '0')
        ret = strobo  # next wave spreaded from center
    return ret

# LC509. Fibonacci Number
def fib(self, n: int) -> int:  # O(n) time and O(1) space
    a, b = 0, 1
    for _ in range(0, n):
        a, b = b, a+b
    return a
def fib(self, n: int) -> int:
    @lru_cache(None)
    def calc(n):
        if n == 0 or n == 1: return n
        return calc(n-1) + calc(n-2)
    return calc(n)

# LC202. Happy Number - digit square sums
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

# LC263. Ugly Number
def isUgly(self, num: int) -> bool:
    for p in 2, 3, 5:
        while num % p == 0 < num:
            num /= p
    return num == 1

# LC402. Remove K Digits - int remove digits to get min
def removeKdigits(self, num: str, k: int) -> str:
    numStack = []
    for digit in num:  # monotone increasing
        while k and numStack and numStack[-1] > digit:
            numStack.pop()
            k -= 1
        numStack.append(digit)
    finalStack = numStack[:-k] if k else numStack
    return "".join(finalStack).lstrip('0') or "0"

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

# LC9. Palindrome Number
def isPalindrome(self, x: int) -> bool:  # O(logn)
    if x < 0 or (x > 0 and x % 10 == 0): return False  # 1 and 10 returns same in below
    rev = 0
    while x > rev:
        rev = rev * 10 + x % 10
        x = x // 10
    return x == rev or x == rev // 10

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





# LC343. Integer Break - 2 ** 3 < 3 ** 2, use derivative to know max x = e, break integer to factors
def integerBreak(self, n: int) -> int:
    if n == 2 or n == 3: return n - 1
    if n % 3 == 0: return 3**(n//3)
    if n % 3 == 1: return 3**(n//3 - 1)*4
    if n % 3 == 2: return 3**(n//3)*2

# LC1025. Divisor Game # BBG
def divisorGame(self, N: int) -> bool:
    return N & 1 == 0

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
