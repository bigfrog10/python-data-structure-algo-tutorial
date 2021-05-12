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

def isMatch(self, s: str, p: str) -> bool:  # Iterative, very fast
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
        else: # backtrack in * case
            p_idx = star_idx + 1  # when there is a * match
            s_idx = s_tmp_idx + 1
            s_tmp_idx = s_idx
    # The remaining characters in the pattern should all be '*' characters
    return all(x == '*' for x in p[p_idx:])
