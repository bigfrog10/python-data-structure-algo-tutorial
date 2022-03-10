# LC415. Add Strings
def addStrings(self, num1: str, num2: str) -> str:
    len1, len2 = len(num1) - 1, len(num2) - 1
    ret, carry = [], 0
    while len1 > -1 or len2 > -1 or carry:
        x1 = int(num1[len1]) if len1 > -1 else 0
        x2 = int(num2[len2]) if len2 > -1 else 0
        sum1 = x1 + x2 + carry
        carry, digit = divmod(sum1, 10)
        ret.append(digit)
        len1 -= 1
        len2 -= 1
    return ''.join(str(d) for d in ret[::-1])

# LC371. Sum of Two Integers
def getSum(self, a: int, b: int) -> int:
    mask = 0xFFFFFFFF  # 32 bit of 1's
    while b != 0:
        a, b = (a ^ b) & mask, ((a & b) << 1) & mask
    max_int = 0x7FFFFFFF
    # ^ mask revert a's bits: with 32 bits, ~ and ^ invert bits twice, so no change
    # outside 32 bits, ~ inverts bits once, so all 0's are now 1's, - for extending negative one's
    return a if a < max_int else ~(a ^ mask)  # ^ mask revert a's bits

# LC67. Add Binary
def addBinary(self, a, b):
    summ, carry = [], 0
    i, j = len(a) - 1, len(b) - 1
    while i >= 0 or j >= 0 or carry:
        d1 = int(a[i]) if i >= 0 else 0
        d2 = int(b[j]) if j >= 0 else 0
        carry, d = divmod(d1 + d2 + carry, 2)
        summ.append(str(d))
        i, j = i-1, j-1
    return "".join(summ[::-1])
def addBinary(self, a: str, b: str) -> str:
    x, y = int(a, 2), int(b, 2)
    while y:
        x, y = x ^ y, (x & y) << 1
    return f'{x:0b}'

# LC29. Divide Two Integers
def divide(self, A, B):
    if A == -2147483648 and B == -1: return 2147483647  # overflow  # -2^31
    a, b, res = abs(A), abs(B), 0
    for x in range(32)[::-1]:
        if (a >> x) - b >= 0: # this is long division from left
            res += 1 << x
            a -= b << x
    return res if (A > 0) == (B > 0) else -res

# LC43. Multiply Strings
def multiply(self, num1, num2):  # O(mn)
    res = [0]* (len(num1) + len(num2))
    for i, e1 in enumerate(reversed(num1)):
        for j, e2 in enumerate(reversed(num2)):
            res[i+j] += int(e1) * int(e2)  # multiply
            res[i+j+1] += res[i+j] // 10  # move carry to next
            res[i+j] %= 10   # remove carry from current
    while len(res) > 1 and res[-1] == 0: res.pop()  # if not used
    return ''.join( map(str,res[::-1]) )
def multiply(self, num1, num2):
    if num1 == '0' or num2 == '0': return '0'
    x, y = int(num1), int(num2)
    ret = cnt = 0
    while x:  # if x & 1 = 0, then this bit is zero, we dont do anything.
        if x & 1 == 1: ret += y << cnt
        cnt += 1
        x = x >> 1
    return str(ret)
