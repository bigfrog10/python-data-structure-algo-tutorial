from collections import defaultdict

def matching_pairs(s, t):
    cnt = 0 # count equal chars
    for i, c in enumerate(s):
        if i < len(t) and t[i] == s[i]:
            cnt += 1
    # swap, this will change cnt by +/- 2
    t_idx = defaultdict(set)  # t's char to indices
    for i, c in enumerate(t):
        t_idx[c].add(i)

    for i, c in enumerate(s):
        if i < len(t) and t[i] == s[i]: continue
        c_idxs = t_idx[c]
        for ci in c_idxs:
            if ci < len(s) and i in t_idx[s[ci]]:
                return cnt + 2
    return cnt - 2

print(matching_pairs('abcd', 'adcb'))
print(matching_pairs('mno', 'mno'))
print(matching_pairs('abcd', 'adcbe'))
print(matching_pairs('abcde', 'adcb'))
