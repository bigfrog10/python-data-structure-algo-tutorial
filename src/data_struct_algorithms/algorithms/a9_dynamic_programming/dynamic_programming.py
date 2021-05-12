

# LC920. Number of Music Playlists
def numMusicPlaylists(self, N, L, K):
    @lru_cache(None)
    def dp(i, j): # num of playlists of length i that has exactly j unique songs
        if i == 0: return +(j == 0)
        ans = dp(i-1, j-1) * (N-j+1) # jth song is new song, N - (j-1) ways
        ans += dp(i-1, j) * max(j-K, 0) # already have j songs, wait K
        return ans % (10**9+7)
    return dp(L, N)

# LC62. Unique Paths
def uniquePaths(self, m, n):
    if not m or not n: return 0
    cur = [1] * n
    for i in range(1, m):
        for j in range(1, n): cur[j] += cur[j-1]  # previous cur[j] + current cur[j-1]
    return cur[-1]

# LC1000. Minimum Cost to Merge Stones
def mergeStones(self, stones: List[int], K: int) -> int:
    if not stones: return 0
    if K == 0: return 0
    n = len(stones)
    if (n - 1) % (K - 1) != 0: return -1
    cusum = [0] + list(accumulate(stones))
    @functools.lru_cache(None)
    def dp(i, j):  # min cost of merging stones[i:j+1]
        minc = float('inf')
        for m in range(j, i, -K+1):  # from high to low reduce K-1 piles at a time.
            t = dp(i, m-1) + dp(m, j)
            minc = min(minc, t)
        # we only add cost when it's doable
        if (j - i) % (K-1) == 0: minc += cusum[j+1] - cusum[i]
        return 0 if minc == float('inf') else minc
    return dp(0, n-1)

# LC221. Maximal Square
from collections import defaultdict
from itertools import product
def maximalSquare(self, matrix: List[List[str]]) -> int: # DP
    if not matrix: return 0
    rows, cols = len(matrix), len(matrix[0])
    # DP(i, j) is the largest side of all squares ended at (i, j)
    dp = defaultdict(int)
    max_len = 0 # track this
    for i, j in product(range(rows), range(cols)):
        if matrix[i][j] == '1':
            dp[i+1, j+1] = min([dp[i+1, j], dp[i, j+1], dp[i, j]]) + 1
            max_len = max(max_len, dp[i+1, j+1])
    return max_len ** 2
