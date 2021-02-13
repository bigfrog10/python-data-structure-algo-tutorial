

def primes_upto(n: int) -> int:
    import math

    if n <= 2:
        return 0

    if n == 3:
        return 1

    flag = [1] * n  # assume all primes
    flag[0] = flag[1] = 0  # non prime
    upper = math.ceil(math.sqrt(n))
    for i in range(2, upper+1):
        if flag[i] == 0:  # non prime
            continue
        else:
            m = (n - i - 1) // i  # every step with i size, how may steps
            if 2 * i < n:
                flag[2*i::i] = [0] * m

    return sum(flag)


print(primes_upto(0))
print(primes_upto(1))
print(primes_upto(2))
print(primes_upto(3))
print(primes_upto(4))
print(primes_upto(5))
print(primes_upto(6))
print(primes_upto(7))
print(primes_upto(8))
print(primes_upto(9))
print(primes_upto(10))
