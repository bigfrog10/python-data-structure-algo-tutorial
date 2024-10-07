
# LC2707. Extra Characters in a String
def minExtraChar(self, s: str, dictionary: List[str]) -> int:
    @cache  # O(n^2) time, O(n + dict) space
    def dfs(s):
        if not s: return 0
        ans = 1 + dfs(s[1:])  # check if 1st char is a leftover
        for i in range(len(s)):
            sub = s[:i+1]
            if sub in words: ans = min(ans, dfs(s[i+1:]))
        return ans
    words = set(dictionary)
    return dfs(s)