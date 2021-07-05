
# LC920. Number of Music Playlists
def numMusicPlaylists(self, N, L, K):
    @lru_cache(None)
    def dp(i, j): # num of playlists of length i that has exactly j unique songs
        if i == 0: return +(j == 0)
        ans = dp(i-1, j-1) * (N-j+1) # jth song is new song, N - (j-1) ways
        ans += dp(i-1, j) * max(j-K, 0) # already have j songs, wait K
        return ans % (10**9+7)
    return dp(L, N)


