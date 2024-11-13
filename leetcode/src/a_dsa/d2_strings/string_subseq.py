
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

# LC1143. Longest Common Subsequence - lcs long common subseq
def longestCommonSubsequence(self, text1: str, text2: str) -> int: # cached recursion
    @lru_cache(None)  # O(nm) time and space
    def solve(i, j):
        if i == len(text1) or j == len(text2): return 0
        if text1[i] == text2[j]: return 1 + solve(i+1, j+1)
        else: return max(solve(i+1, j), solve(i, j+1))
    print(solve.cache_info())
    return solve(0, 0)