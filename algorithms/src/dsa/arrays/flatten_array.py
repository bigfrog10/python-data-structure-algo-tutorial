
def flatten(arr: list, n):
    if n == 0: return arr
    res = []
    for a in arr:
        if isinstance(a, list):
            b = flatten(a, n-1)
            res.extend(b)
        else: res.append(a)
    return res

m = [1, 2, 3, [4, 5, 6], [7, 8, [9, 10, 11], 12], [13, 14, 15]]
print(flatten(m, 0))
print(flatten(m, 1))
print(flatten(m, 2))


s = "add"
from collections import Counter
a = sorted(Counter(s).values())
print(a)
print(a == [1, 2])

import string
print(string.ascii_lowercase)

import itertools
s = '11222333aaa'
for digit, group in itertools.groupby(s):
    print(f'{digit} {list(group)}')

import math
print(math.comb(5, 2))

import functools
print(list(itertools.chain('0', 'abc')))

a = 3
b = 0
print((a, b, b and a/b))

t = 'ahbgdc'
it = iter(t)
for c in 'abc':
    print(c in it)