
# LC947. Most Stones Removed with Same Row or Column
def removeStones(self, stones: List[List[int]]) -> int:
    f = {}  # O(n)
    OFFSET = 10001  # or max(x) + 1
    def find(x):
        fx = f.get(x)
        if fx is None:
            f[x] = x
            return x
        if fx != x: f[x] = fx = find(fx)
        return fx
    def merge(u, v):
        fu, fv = find(u), find(v)
        if fu != fv: f[fu] = fv
    for u, v in stones: merge(u, OFFSET+v)
    ans = 0
    for k in f.keys():
        if find(k) == k: ans += 1
    return len(stones) - ans
# https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/?envType=company&envId=apple&favoriteSlug=apple-more-than-six-months
