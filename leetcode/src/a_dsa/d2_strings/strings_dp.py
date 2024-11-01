
# LC2707. Extra Characters in a String
def minExtraChar(self, s: str, dictionary: List[str]) -> int:
    n = len(s)  # O(n^2 + build trie)
    root = self.buildTrie(dictionary)
    @cache
    def dp(start):  # number of extra chars in string starting from start
        if start == n: return 0
        ans = dp(start + 1) + 1  # check if 1st char is a leftover
        node = root
        for end in range(start, n):
            if s[end] not in node: break
            node = node[s[end]]
            if '$' in node:
                ans = min(ans, dp(end + 1))
        return ans
    return dp(0)
def buildTrie(self, dictionary):
    root = {}
    for w in dictionary:
        node = root
        for c in w: node = node.setdefault(c, {})
        node['$'] = True
    return root