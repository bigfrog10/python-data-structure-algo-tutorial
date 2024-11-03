from typing import List
import math

# LC3014. Minimum Number of Pushes to Type Word I phone pad min push distinct
def minimumPushes(self, word: str) -> int:
    p, m = divmod(len(word), 8)
    return 8 * p * (p+1) // 2 + m * (p+1)

# LC3016. Minimum Number of Pushes to Type Word II  phone pad min push
def minimumPushes(self, word: str) -> int:  # O(n) time, O(1) space
    freq = [0] * 26
    for c in word: freq[ord(c) - ord('a')] += 1
    freq.sort(reverse=True)  # constant
    # Find the size of the array without trailing zeroes
    sz = next((i for i, x in enumerate(freq) if x == 0), 26)
    total_pushes = 0  # Calculate the minimum pushes
    for i in range(sz):
        total_pushes += freq[i] * (i // 8 + 1)
    return total_pushes

# LC17. Letter Combinations of a Phone Number - phone letter combo phone pad
def letterCombinations(self, digits):  # Time & Space: O(4^n)
    dict = {'2':"abc", '3':"def",  '4':"ghi", '5':"jkl",
            '6':"mno", '7':"pqrs", '8':"tuv", '9':"wxyz"}
    cmb = [''] if digits else []
    for d in digits: cmb = [p + q for p in cmb for q in dict[d]]
    return cmb

# LC118. Pascal's Triangle
def generate(self, numRows): # Time & Space: O(n^2)
    row, res = [1], []
    for n in range(numRows):
        res.append(row)
        row = [1] + [row[i] + row[i+1] for i in range(n)] + [1]
    return res


# LC78. Subsets  - unique elem
def subsets(self, nums: List[int]) -> List[List[int]]:  # time and space O(2^N)
    if not nums: return []
    ret = [[]]
    for num in nums: ret += [ss + [num] for ss in ret]  # add one digit at a time
    return ret

def subsets(self, nums: List[int]) -> List[List[int]]:  # samiliar to LC90
    n, ans = len(nums), []
    def backTrack(start, cur_list):
        ans.append(cur_list[:])
        for j in range(start, n):
            cur_list.append(nums[j])
            backTrack(j+1, cur_list)
            cur_list.pop()
    backTrack(0, [])
    return ans

# LC90. Subsets II - dupe elem all possible subsets
def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:  # O(n * 2^n)
    nums.sort()
    ans = []
    def generate(i, solution):  # O(n) space from solution
        ans.append(solution[:])  # copy
        for j in range(i, len(nums)):
            # We can re-use numbers, but not at this position and same previous premutation
            if j > i and nums[j] == nums[j-1]:
                continue  # only when i==j, we take dupes
            generate(j+1, solution + [nums[j]])  # go down 1 element and then backout to empty
    generate(0, [])
    return ans  # [[],[1],[1,2],[1,2,2],[2],[2,2]]

# LC46. Permutations
def permute(nums: List[int]) -> List[List[int]]:  # O(n!) & O(n!)
    ans = [[]]
    for num in nums:
        ans = [each[:i] + [num] + each[i:] for each in ans for i in range(len(each)+1)]
    return ans

# LC47. Permutations II  has dupes   perm dupe
def permuteUnique(self, nums: List[int]) -> List[List[int]]:  # O(n!), T(n) = n*T(n-1) + O(n)
    ans = [[]]  # best case is when all numbers are the same, while worst case is when all numbers are distinct
    for n in nums:
        new_ans = []
        for p in ans:
            for i in range(len(p)+1):
                new_ans.append(p[:i] + [n] + p[i:])
                if i < len(p) and p[i] == n: break  # we insert dups only from left side.
        ans = new_ans
    return ans

# LC60. Permutation Sequence - return kth permutation
def getPermutation(self, n: int, k: int) -> str: # O(n^2) due to pop(i)
    nums = list(map(str, range(1, n+1)))  # '1', '2', ..., 'n'
    fact = math.factorial(len(nums)-1)
    k, ans = k-1, ''  # zero based
    while k:
        i, k = divmod(k, fact)  # there are (n-1)! buckets in n!, we want to know which one
        ans += nums.pop(i)  # it's ith bucket starting with i, so pop i
        fact //= len(nums)  # next bucket size is (n-2)! = (n-1)! / (n-1)
    ans += ''.join(nums)
    return ans

# LC31. Next Permutation
def nextPermutation(self, nums: List[int]) -> None:  # O(n)
    if not nums: return
    n = len(nums)
    if n == 1: return
    # from back, find the first value such that value < right, 1, 8, 4, 7, 6, 5, 3, 1 => 4 (idx=2)
    idx = next((i-1 for i in range(n)[::-1] if nums[i-1] < nums[i]), -1)
    if idx == -1: nums.reverse()
    else:
        # find the value such that prev > value > next, find 5 (idx=5)
        idx1 = next((i-1 for i in range(idx+1, n) if nums[i] <= nums[idx]), n-1)
        nums[idx], nums[idx1] = nums[idx1], nums[idx]  # swap
        nums[idx+1:] = reversed(nums[idx+1:])

# LC1053. Previous Permutation With One Swap    prev permutation
def prevPermOpt1(self, arr: List[int]) -> List[int]: # Time: O(n) Space: O(1)
    nums = arr.copy()
    if not arr: return []
    n = len(nums)
    idx = next((i-1 for i in range(n)[::-1] if nums[i-1] > nums[i]), -1)
    if idx == -1: return nums
    idx1 = next((i-1 for i in range(idx+1, n) if nums[i] > nums[idx]),
                n-1 if nums[idx] > nums[n-1] else idx+1)
    # 2 cases for above: [1,9,4,6,7] -> [1,7,4,6,9],  [3,1,1,3] -> [1,3,1,3]
    nums[idx], nums[idx1] = nums[idx1], nums[idx]
    return nums

    #
    n = len(arr)  # [1,9,4,6,7]
    idx = n-2  # looking for the first rise index inside from right
    while idx >= 0 and arr[idx] <= arr[idx+1]: idx -= 1  # idx = 1 for 9 > 4
    if idx >= 0:  # Otherwise, we have increasing series, just return
        midx = idx + 1  # now find max < arr[idx], swap with that
        for i in range(midx, n):  # max != arr[idx], otherwise no change in swap
            if arr[idx] > arr[i]:
                if arr[i] > arr[midx]: midx = i  # midx=4 for 9 is max so hit end
            else: break
        arr[idx], arr[midx] = arr[midx], arr[idx]
    return arr

# LC77. Combinations  combos
def combine(self, n, k):
    return list(itertools.combinations(range(1, n+1), k))
# 4, 3: [[3],[4]], then  [[2,3],[2,4],[3,4]], and then [[1,2,3],[1,2,4],[1,3,4],[2,3,4]]
def combine(self, n: int, k: int) -> List[List[int]]:  # O(k * C^k_n)
    combs = [[]]
    for j in range(1, k+1)[::-1]:  # backward is much faster
        combs = [[i] + c for c in combs for i in range(j, c[0] if c else n+1)]
    return combs
def combine(self, n: int, k: int) -> List[List[int]]:
    combs = [[]]
    for _ in range(k):
        combs = [[i] + c for c in combs for i in range(1, c[0] if c else n+1)]
    return combs

# for 4, 3: [[1],[2],[3],[4]],  [[1,2],[1,3],[2,3],[1,4],[2,4],[3,4]]
# and  [[1,2,3],[1,2,4],[1,3,4],[2,3,4]]
def combine(self, n: int, k: int) -> List[List[int]]:  # O(k * C^k_n)
    nums = list(range(1, k + 1)) + [n + 1]  # init first combination
    output, j = [], 0
    while j < k:
        output.append(nums[:k])
        # increase first nums[j] by one
        # if nums[j] + 1 != nums[j + 1]
        j = 0
        while j < k and nums[j + 1] == nums[j] + 1:
            nums[j] = j + 1
            j += 1
        nums[j] += 1
    return output

def combine(self, n: int, k: int) -> List[List[int]]:
    def backtrack(curr, first_num):
        if len(curr) == k:
            ans.append(curr[:])
            return
        need = k - len(curr)
        remain = n - first_num + 1
        available = remain - need
        for num in range(first_num, first_num + available + 1):
            curr.append(num)
            backtrack(curr, num + 1)
            curr.pop()
    ans = []
    backtrack([], 1)
    return ans

# LC920. Number of Music Playlists
def numMusicPlaylists(self, N, L, K):
    @lru_cache(None)
    def dp(i, j): # num of possible playlists of length i that has exactly j unique songs
        if i == 0:
            return +(j == 0)
        ans = dp(i-1, j-1) * (N-j+1) # N-j+1: the number of new songs we can add to the playlist
        ans += dp(i-1, j) * max(j-K, 0) # number of old songs we can play
        return ans % (10**9+7)
    return dp(L, N)

# LC216. Combination Sum III 1 through 9 sum to target
from itertools import combinations
def combinationSum3(self, k: int, n: int) -> List[List[int]]:
    return [c for c in combinations(range(1, 10), k) if sum(c) == n]
def combinationSum3(self, k: int, n: int) -> List[List[int]]: # O(9!*k/(9-k)!), c(9, k)
    results = []  # space O(k)
    def backtrack(remain, comb, next_start):
        if remain == 0 and len(comb) == k:
            results.append(list(comb))
            return
        elif remain < 0 or len(comb) == k: return
        for i in range(next_start, 9):
            comb.append(i + 1)
            backtrack(remain - i - 1, comb, i + 1)
            comb.pop()  # backtrack the current choice
    backtrack(n, [], 0)
    return results




















