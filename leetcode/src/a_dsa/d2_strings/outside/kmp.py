# copied from: https://towardsdatascience.com/pattern-search-with-the-knuth-morris-pratt-kmp-algorithm-8562407dba5b
# This seems reasonable with least technical maneuver.
# Good example: https://web.stanford.edu/class/cs97si/10-string-algorithms.pdf
# This has a lot of other algorithms.
# https://www.youtube.com/watch?v=EL4ZbRF587g
# https://www.toutiao.com/article/7100531067098153502/

def kmp_pi(s: str):
    # return the length of the longest *proper* prefix of s which is also a suffix.
    # note that s itself satisfies this condition but is not proper/substring.
    m = len(s)
    pi = [0] * m
    fmm_idx = 0  # index of first mismatch
    for i in range(1, m):
        while fmm_idx and s[i] != s[fmm_idx]:
            # when there is a mismatch, we try previous matches.
            fmm_idx = pi[fmm_idx-1]

        # if fmm_idx = 0, continue with pi[i] = 0

        if s[i] == s[fmm_idx]:  # if matched, move forward
            fmm_idx += 1
            pi[i] = fmm_idx

    return pi


print(kmp_pi('ababaca'))  # [0, 0, 1, 2, 3, 0, 1]
print(kmp_pi('aaa'))  # [0, 1, 2]
print(kmp_pi('abc'))  # [0, 0, 0]


def kmp_search(text, pattern):
    # return first indices of all occurrences of pattern in text
    matches = []

    pi = kmp_pi(pattern)

    n, m = len(text), len(pattern)
    k = 0  # pattern index
    for i in range(n):
        while k and text[i] != pattern[k]:
            k = pi[k - 1]

        # if k = 0, continue

        if text[i] == pattern[k]:
            if k == m - 1:
                matches.append(i - k)
                k = pi[k]
            else:
                k += 1

    return matches


print(kmp_search('acfacabacabacacdk', 'acabacacd'))  # [7]
print(kmp_search('abcdabcdeabc', 'abc'))  # [0, 4, 9]
print(kmp_search('adabeabafababgababc', 'ababc'))
