
# cumulative counts on consonants and vowels,
def cumu_counts(string: str):  # O(n)
    vowels = {'a', 'e', 'i', 'o', 'u'}

    csum = 1 if string[0] in vowels else 0
    vsum = 1 - csum
    c_cnts, v_cnts = [csum], [vsum]
    for i in range(1, len(string)):
        csum += 1 if string[i] in vowels else 0
        vsum += 1 if string[i] in vowels else 0
        c_cnts.append(csum)
        v_cnts.append(vsum)

    return c_cnts, v_cnts


def imbalance(string: str) -> int:
    vowels = {'a', 'e', 'i', 'o', 'u'}

    def dfs(s):
        if not s:  # base cases
            return 0

        if len(s) == 1:
            return 1

        # if we want to remove this loop, we need to track begin/end of s.
        # then use above cumulative sum
        count = 0
        for c in s:
            if c in vowels:
                count += 1

        n = len(s)
        # count: vowels, n - count: consonants, so diff = n-count - count
        res = abs(n - count - count) + dfs(s[:n // 2]) + dfs(s[n // 2:])
        return res

    return dfs(string)


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
