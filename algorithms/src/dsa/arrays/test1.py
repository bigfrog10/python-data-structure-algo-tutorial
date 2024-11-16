import collections
import itertools

ransomNote = "aa"
magazine = "aba"
print(collections.Counter(ransomNote) - collections.Counter(magazine))
print(not collections.Counter(ransomNote) - collections.Counter(magazine))

# print(sum(s for s in ['a', 'b']))

x = ['a', 'b']
print(map(x.index, x))

nums = [1,2,1,1,2,1,2]
for k, v in itertools.groupby(nums):
    for w in v:
        print(f'{k}: {w}')
    print(' ')
# print(list(itertools.groupby(nums)))

s = 'ab12345'
for prev_ch, ch in itertools.pairwise(itertools.chain('0', s)):
    print(f'{prev_ch}: {ch}')

print(list(itertools.chain('0', 'abc')))

x = 3
y = 7
print(bin(x^y))

print(int("0", 16))

b = "1110"
print(b.count('0') * b.count('1') )

# print(b & (-b))

a = [(1, 2), (3, 4)]
import bisect
i = bisect.bisect(a, (1,))
print(i)

for i in range(5, 3, -1):
    print(f'i={i}')