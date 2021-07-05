
# LC10. Regular Expression Matching, top100
def isMatch(self, s: str, p: str) -> bool:  # recursive, slow
    if not p: return not s
    first_match = s and p[0] in {s[0], '.'} # "*" can't be the first one
    if len(p) > 1 and p[1] == '*':
        return self.isMatch(s, p[2:]) or first_match and self.isMatch(s[1:], p)
    else: return first_match and self.isMatch(s[1:], p[1:])

# LC44. Wildcard Matching
import functools
def isMatch(self, s: str, p: str) -> bool:
    @functools.lru_cache(maxsize=None)
    def walk(s, p): # DFS
        if s == p: return True # typically "" == ""
        if not s: return p == '*' * len(p) # all *
        if not p: return False # one of them is empty, the other not.
        if p[0] == '*': return walk(s, p[1:]) or walk(s[1:], p)
        else: return p[0] in {s[0], '?'} and walk(s[1:], p[1:])
    return walk(s, p)

# replace wildcards in a given binary string with 0 & 1 and produce all combinations
def generate(pattern: str):
    res = []
    def recurse(strlist, idx):
        if idx == len(strlist):
            res.append(''.join(strlist))
        else:
            if strlist[idx] == '?':
                for ch in '01':
                    strlist[idx] = ch
                    recurse(strlist, idx+1)
            else:
                recurse(strlist, idx+1)

    recurse(list(str), 0)
    return rest
