
# LC115. Distinct Subsequences
def numDistinct(self, s: str, t: str) -> int:  # O(n*m) time space
    @functools.cache
    def dp(i, j): # count s[i:] for t[j:]
        if j == len(t): return 1 # match t, has a match
        if i == len(s): return 0 # end of s, no match
        count = dp(i+1, j) # skip this char
        if s[i] == t[j]: count += dp(i+1, j+1)
        return count
    return dp(0, 0)