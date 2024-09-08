
# LC291. Word Pattern II - if bijective mapping existing
def wordPatternMatch(self, pattern: str, s: str) -> bool:  # n*C(n, m) or O(n^m)
    mapping = {}

    def check(pidx, sidx):
        if pidx == len(pattern) and sidx == len(s): return True
        if pidx == len(pattern) or sidx == len(s): return False

        char = pattern[pidx]
        for i in range(sidx, len(s)):
            subs = s[sidx:i+1]
            if mapping.get(char) == subs:
                if check(pidx+1, i+1): return True
            if char not in mapping and subs not in mapping.values():
                mapping[char] = subs
                if check(pidx+1, i+1): return True
                del mapping[char]  # backtrack

        return False
    return check(0, 0)

# LC290. Word Pattern - pattern char to word
def wordPattern(self, pattern: str, string: str) -> bool:
    s = pattern
    t = string.split()
    return len(set(zip(s, t))) == len(set(s)) == len(set(t)) and len(s) == len(t)
