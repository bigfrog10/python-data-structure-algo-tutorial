
# LC3040. Maximum Number of Operations With the Same Score II
def maxOperations(self, nums: List[int]) -> int:  # O(n^2) time and space
    @lru_cache(None)
    def dfs(score: int, lo = 0, hi = len(nums) - 1) -> int:  # operations with this score
        if hi <= lo: return 0
        max_ops1 = 1 + dfs(score, lo + 2, hi) if score == sum(nums[lo : lo + 2]) else 0
        max_ops2 = 1 + dfs(score, lo, hi - 2) if score == sum(nums[hi - 1 : hi + 1]) else 0
        max_ops3 = 1 + dfs(score, lo + 1, hi - 1) if score == nums[lo] + nums[hi] else 0
        return max(max_ops1, max_ops2, max_ops3)
    return max(dfs(sum(nums[: 2])), dfs(sum(nums[-2 :])), dfs(nums[0] + nums[-1]))