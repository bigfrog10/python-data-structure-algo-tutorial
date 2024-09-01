
# LC10. Regular Expression Matching, top100 - * repeating
def isMatch(self, text, pattern): # O(nm)
    @functools.lru_cache(None)
    def dp(i, j):
        if j == len(pattern): ans = i == len(text)
        else:
            first_match = i < len(text) and pattern[j] in {text[i], '.'}
            if j+1 < len(pattern) and pattern[j+1] == '*':
                ans = dp(i, j+2) or first_match and dp(i+1, j)
            else: ans = first_match and dp(i+1, j+1)
        return ans
    return dp(0, 0)
@functools.lru_cache(None)
def isMatch(self, s: str, p: str) -> bool:  # O(len(s)*len(p)) due to dp, but substring take len(s) and len(p)
    if not p: return not s
    first_match = s and p[0] in {s[0], '.'}  # "*" can't be the first one
    if len(p) > 1 and p[1] == '*':
        return self.isMatch(s, p[2:]) or first_match and self.isMatch(s[1:], p)
    else: return first_match and self.isMatch(s[1:], p[1:])
# slower
@lru_cache(None)
def isMatch(self, s: str, p: str) -> bool:  # O(len(s)*len(p)) due to dp, but substring take len(s) and len(p)
    if not p: return not s
    first_match = s and p[0] in {s[0], '.'}  # "*" can't be the first one
    if len(p) > 1 and p[1] == '*':
        return self.isMatch(s, p[2:]) or first_match and self.isMatch(s[1:], p)
    else: return first_match and self.isMatch(s[1:], p[1:])

# LC44. Wildcard Matching - * is any sequence
def isMatch(self, s: str, p: str) -> bool:  # O(mn)
    @functools.lru_cache(maxsize=None)
    def walk(i, j):  # DFS, O(mn)
        if i == len(s) and j == len(p): return True  # end
        if i == len(s): return p[j:] == '*' * (len(p) - j)
        if j == len(p): return False
        if p[j] == '*': return walk(i, j+1) or walk(i+1, j)
        else: return p[j] in {s[i], '?'} and walk(i+1, j+1)
    return walk(0, 0)
def isMatch(self, s: str, p: str) -> bool:  # Iterative, very fast, O(mn)
    sl, pl = len(s), len(p)
    s_idx = p_idx = 0  # 2 walkers on s and p
    star_idx = s_tmp_idx = -1  # 2 markers
    while s_idx < sl:
        if p_idx < pl and p[p_idx] in {s[s_idx], '?'}:
            s_idx += 1
            p_idx += 1
        elif p_idx < pl and p[p_idx] == '*':
            star_idx = p_idx  # save current states for backtrack
            s_tmp_idx = s_idx
            p_idx += 1  # Then try first of 2 cases, match 0 char, skip *
        elif star_idx == -1:  # If there is no * in previous and no match currently
            return False
        else: # backtrack in * case, start_idx != -1
            p_idx = star_idx + 1  # skip * and match 1 char next line, fixed
            s_tmp_idx = s_idx = s_tmp_idx + 1  # keep going for all unmatched chars
    # The remaining characters in the pattern should all be '*' characters
    return all(x == '*' for x in p[p_idx:])

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
