
# LC1718. Construct the Lexicographically Largest Valid Sequence   lexi large seq lexi seq lexico largest
def constructDistancedSequence(self, n: int) -> List[int]:  # O(n!) time, O(n) space
    m = 2*n - 1
    A = [0] * m  # 0 means not
    visited = set()
    def dfs(i):  # i is position in the array
        if i == m: return all(A)  # all assigned or not
        if A[i]: return dfs(i+1)  # occupied, move to next position
        for v in range(n, 0, -1):  # position from back
            j = i if v == 1 else i + v  # there is only one 1, leave it last
            if j < m and not A[j] and v not in visited:
                A[i] = A[j] = v
                visited.add(v)
                if dfs(i+1): return True
                A[i] = A[j] = 0  # backout
                visited.remove(v)
        return False
    if dfs(0): return A
    return []

# LC3040. Maximum Number of Operations With the Same Score II  max ops
def maxOperations(self, nums: List[int]) -> int:  # O(n^2) time and space
    @lru_cache(None)
    def dfs(score: int, lo = 0, hi = len(nums) - 1) -> int:  # operations with this score
        if hi <= lo: return 0
        max_ops1 = 1 + dfs(score, lo + 2, hi) if score == sum(nums[lo : lo + 2]) else 0
        max_ops2 = 1 + dfs(score, lo, hi - 2) if score == sum(nums[hi - 1 : hi + 1]) else 0
        max_ops3 = 1 + dfs(score, lo + 1, hi - 1) if score == nums[lo] + nums[hi] else 0
        return max(max_ops1, max_ops2, max_ops3)
    return max(dfs(sum(nums[: 2])), dfs(sum(nums[-2 :])), dfs(nums[0] + nums[-1]))

# LC996. Number of Squareful Arrays
def numSquarefulPerms(self, nums: List[int]) -> int:  # O(n!) time, O(n) space LC!
    self.count = 0  # space should be O(n^2) because we copy arrays
    def dfs(arr, path):
        if not arr: self.count += 1
        for i in range(len(arr)):
            if i and arr[i] == arr[i-1]: continue # remove dupe evaluation
            if not path or math.sqrt(path[-1] + arr[i]).is_integer():
                dfs(arr[:i] + arr[i+1:], path + [arr[i]])
    dfs(sorted(nums), [])
    return self.count

# LC2305. Fair Distribution of Cookies
def distributeCookies(self, cookies: List[int], k: int) -> int:
    def dfs(p):  # O(kn) time, O(k+n) space
        nonlocal best
        if p==len(cookies):
            best = min(best, max(split))
            return
        # give to a new kid
        if len(split)<k:
            split.append(cookies[p])
            dfs(p+1)
            split.pop()
        # give to a kid that already has cookies
        for i in range(len(split)):
            if split[i]+cookies[p] < best:
                split[i] += cookies[p]
                dfs(p+1)
                split[i] -= cookies[p]
    split = []
    best = float("inf")
    dfs(0)
    return best
# https://leetcode.com/problems/fair-distribution-of-cookies/solutions/3634043/python-efficient-backtracking-fully-explained/?envType=company&envId=amazon&favoriteSlug=amazon-three-months

# LC403. Frog Jump
def canCross(self, stones: List[int]) -> bool:  # O(n^2)
    @cache
    def dfs(i, k):
        if i == len(stones) - 1: return True
        res = False
        for j in range(i + 1, len(stones)):
            if stones[i] + k + 1 < stones[j]: break
            if stones[i] + k == stones[j]:
                res = res or dfs(j, k)
            if stones[i] + k + 1 == stones[j]:
                res = res or dfs(j, k + 1)
            if stones[i] + k - 1 == stones[j]:
                res = res or dfs(j, k - 1)
        return res
    return dfs(0, 0)











