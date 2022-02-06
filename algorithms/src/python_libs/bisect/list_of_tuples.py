import bisect
a = [(1, 3), (4, 5), (6, 7), (9, 10)]
idx = bisect.bisect(a, (2, 2))
print(idx)
