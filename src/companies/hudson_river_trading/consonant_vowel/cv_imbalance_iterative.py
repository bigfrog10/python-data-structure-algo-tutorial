
def imbalance(string: str) -> int:
    vowels = {'a', 'e', 'i', 'o', 'u'}

    ret = 0
    stack = [string]
    while stack:
        s = stack.pop()
        if len(s) == 1:
            ret += 1
            continue

        count = 0
        for c in s:
            if c in vowels:
                count += 1

        n = len(s)
        ret += abs(n - 2 * count)
        if s[:n // 2]:
            stack.append(s[:n // 2])
        if s[n // 2:]:
            stack.append(s[n // 2:])

    return ret


print(imbalance(''))  # 0
print(imbalance('a'))  # 1
print(imbalance('b'))  # 1
print(imbalance('aa'))  # 4 -> 2 from aa, 2 ones from each a
print(imbalance('bb'))  # 4
print(imbalance('ab'))  # 2 -> 1 from a, 1 from b
print(imbalance('aaa'))  # 8 -> 3 from aaa, 1 a and 2 a, then a and a
print(imbalance('aab'))  # 4 -> 1 from aab, 1 from a, 2 from ab
print(imbalance('abb'))  # 6 -> 1 from abb, 1 from a, 4 from bb
print(imbalance('bbb'))  # 8
print(imbalance('aeiou'))  # 17 -> 5 from aeiou, 4 from ae, 8 from iou
print(imbalance('pqrst'))
