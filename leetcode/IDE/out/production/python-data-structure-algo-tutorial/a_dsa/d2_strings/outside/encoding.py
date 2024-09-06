# LeetCode 471

import functools


@functools.lru_cache(None)
def abbreviate(string):
    L = len(string)
    res = string
    for k in range(1, 1 + (L // 2)):
        if not L % k:                                             # Is abbreviation the right length? i.e. if string has 10 chars cand must have 1, 2, or 5 chars
            cand = string[:k]                                     # Candidate for abbreviation
            if string == (L // k) * cand:                         # cand = 'abc' string = 'abcabcabc' then cand is a valid abbreviation of s
                cand = min(cand, helper(0, 1, cand), key=len)   # See if we can shorten cand even more i.e. 2[abbbabbbc] shortened to 2[2[abb]c]
                res = min(res, str(L//k)+'['+cand+']', key=len)
    return res

@functools.lru_cache(None)
def helper(i, j, s):
    if j == len(s):  # If j (the right pointer) equals len(s) then return s[i:j] or abbreviate(s[i:j]) - whichever is shorter
        return min(s[i:j], abbreviate(s[i:j]), key=len)
    return min(helper(i, j+1, s), abbreviate(s[i:j]) + helper(j, j+1, s), key=len) # Choose to abbreviate s[i:j] or not to abbreviate s[i:j]


print(helper(0, 1, "abbbabbbcabbbabbbc"))  # "2[2[abbb]c]"

@functools.lru_cache(None)
def encode(self, s: str) -> str:
    if len(s) < 5: return s
    i = (s+s).find(s,1)
    encoded = str(len(s)//i) + '[' + self.encode(s[:i]) + ']' if i<len(s) else s
    splitEncoded = [self.encode(s[:i]) + self.encode(s[i:]) for i in range(1,len(s))]
    return min(splitEncoded + [encoded], key=len)
