
# LC1010. Pairs of Songs With Total Durations Divisible by 60
def numPairsDivisibleBy60(self, time: List[int]) -> int:  # O(n) time and O(1) space
    counts = [0] * 60  # O(1) space
    ret = 0
    for t in time:
        t %= 60
        if t == 0: ret += counts[0]
        else: ret += counts[60-t]
        counts[t] += 1
    return ret

# LC724. Find Pivot Index - break array to left and right with equal sum
def pivotIndex(self, nums: List[int]) -> int:
    S, leftsum = sum(nums), 0
    for i, x in enumerate(nums):
        if leftsum == (S - leftsum - x): return i
        leftsum += x
    return -1

# LC1991. Find the Middle Index in Array - same as above
def findMiddleIndex(self, nums: List[int]) -> int:
    S, leftsum = sum(nums), 0
    for i, x in enumerate(nums):
        if leftsum == (S - leftsum - x): return i
        leftsum += x
    return -1

# LC39. Combination Sum  # O(n^(target/min)) - return combos, can use elems multiple times comb sum
def combinationSum(self, candidates, target):  # fastest
    candidates = sorted(candidates)  # small to large, cut branches
    result = []  # O(n^(target/min cand), power is tree depth
    def dfs(remain, stack): # stack is history
        if remain == 0:
            result.append(stack)
            return
        elif remain < 0: return
        for item in candidates:
            if item > remain: break
            if stack and item < stack[-1]: continue  # smallers are done already.
            else: dfs(remain - item, stack + [item])
    dfs(target, [])
    return result

# LC40. Combination Sum II - could have dupes and use only once
def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    result = [] # O(2^n)
    def combine_sum_2(start, path, target):
        if not target:
            result.append(path)
            return
        for i in range(start, len(candidates)):
            # ignore duplicates
            if i > start and candidates[i] == candidates[i - 1]: continue
            if candidates[i] > target: break # so sum > target, not a solution
            # we used i here, so next search starting from i+1
            combine_sum_2(i + 1, path + [candidates[i]], target - candidates[i])
    candidates.sort()
    combine_sum_2(0, [], target)
    return result

# LC377. Combination Sum IV - return number of combinations
def combinationSum4(self, nums: List[int], target: int) -> int:  # O(T * N)
    @functools.lru_cache(maxsize = None)
    def combs(remain):  # O(target)
        if remain == 0: return 1
        result = 0
        for num in nums:  # O(n)
            if remain - num >= 0: result += combs(remain - num)
        return result
    return combs(target)

# LC698. Partition to K Equal Sum Subsets
def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
    nums, ASum = [n for n in nums if n != 0], sum(nums) # O(2^n)
    if ASum % k != 0: return False
    subsum, buckets = ASum // k, [0] * k
    nums.sort(reverse=True)  # larger number first
    lnth = len(nums)
    def dfs(i):  #DFS determines which bucket to put the 'current element' (nums[idx] ) into
        if i == lnth: return True
        for j in range(k): # For each bucket
            buckets[j] += nums[i]
            if buckets[j] <= subsum and dfs(i+1): # if we successfully place ith element
                return True
            buckets[j] -= nums[i] # back out
            # if we put ith into one empty bucket unsucessfully,
            # no need to try other empty buckets. simply return False at the end.
            if buckets[j] == 0: break
        return False
    return dfs(0)

# LC713. Subarray Product Less Than K - return # of such subarrays
def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
    if k <= 1: return 0 # nums are positive, so there is no solution
    prod = 1  # watch out for k=0 and k=1, array [0, 1], [1, 1]
    ans = left = 0
    for right, val in enumerate(nums):
        prod *= val
        while prod >= k:
            prod //= nums[left]
            left += 1
        ans += right - left + 1  # all subarrasy ends at nums[right]
    return ans

# LC1480. Running Sum of 1d Array
def runningSum(self, nums: List[int]) -> List[int]:
    output = [nums[0]]
    for i in range (1, len(nums)):
        nums[i] += nums[i - 1]
        output.append(nums[i])
    return output

# LC1588. Sum of All Odd Length Subarrays
def sumOddLengthSubarrays(self, arr: List[int]) -> int:
    n = len(arr)
    res = 0
    for j in range(1, n + 1, 2):  # O(n^3)
        for i in range(n - j + 1):
            res += sum(arr[i:i + j])
    return res
def sumOddLengthSubarrays(self, A):  # O(n)
    # for any i there are i+1 options on left, n-i options on right.
    # [a[i]] is counted on left, all a's counted on right.
    # for length k arrays, there are (k+1) // 2 odd length arrays
    res, n = 0, len(A)
    for i, a in enumerate(A):
        res += ((i + 1) * (n - i) + 1) // 2 * a  # +1 for ceiling
    return res


