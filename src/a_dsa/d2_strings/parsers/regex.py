
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
    def walk(s, p):  # DFS
        if s == p: return True  # typically "" == ""
        if not s: return p == '*' * len(p)  # all *
        if not p: return False  # s has leftover not matched
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

# LC726. Number of Atoms
def countOfAtoms(self, formula):
    parse = re.findall(r"([A-Z][a-z]*)(\d*)|(\()|(\))(\d*)", formula)
    stack = [collections.Counter()]
    for name, m1, left_open, right_open, m2 in parse:
        if name: stack[-1][name] += int(m1 or 1)
        if left_open: stack.append(collections.Counter())
        if right_open:
            top = stack.pop()
            for k in top: stack[-1][k] += top[k] * int(m2 or 1)
    return "".join(name + (str(stack[-1][name]) if stack[-1][name] > 1 else '')
                   for name in sorted(stack[-1]))
