
# LC46. Permutations
def permute(self, nums: List[int]) -> List[List[int]]:
    ans = [[]]
    for num in nums:
        new_arr = []
        for each in ans:
            for i in range(len(each)+1):
                new_arr.append(each[:i] + [num] + each[i:])
        ans = new_arr
    return ans

# LC17. Letter Combinations of a Phone Number, top100
def letterCombinations(self, digits):
    dict = {'2':"abc", '3':"def",  '4':"ghi", '5':"jkl",
            '6':"mno", '7':"pqrs", '8':"tuv", '9':"wxyz"}
    cmb = [''] if digits else []
    for d in digits: cmb = [p + q for p in cmb for q in dict[d]]
    return cmb

# LC31. Next Permutation
# LC556. Next Greater Element III
def nextPermutation(self, nums: List[int]) -> None:
    if not nums: return
    n = len(nums)
    if n == 1: return
    idx = -1  # find the first value such that value < next
    for i in range(n-1, 0, -1):
        if nums[i-1] < nums[i]:
            idx = i-1
            break
    if idx == -1:
        nums.sort()
        return
    idx1 = n-1  # find the value such that prev > value > next
    for i in range(idx+1, n):
        if nums[i] <= nums[idx]:
            idx1 = i-1
            break
    nums[idx], nums[idx1] = nums[idx1], nums[idx]
    for i in range(idx+1, (n + idx+1)// 2): # reverse after idx: we swap only half, otherwise we swap twice
        nums[i], nums[n-i+idx] = nums[n-i+idx], nums[i]



# LC47. Permutations II
def permuteUnique(self, nums): # best
    ans = [[]]
    for n in nums:
        new_ans = []
        for p in ans:
            for i in range(len(p)+1):
                new_ans.append(p[:i] + [n] + p[i:])
                # we insert dups only from left side.
                if i < len(p) and p[i] == n: break
        ans = new_ans
    return ans
def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    n = len(nums)
    visited = set()
    def backtrack(idx):
        if idx == n: visited.add(tuple(nums))
        for i in range(idx, n):
            # remove dups
            if i != idx and nums[idx] == nums[i]: continue
            # print(f'i={i}, idx={idx}')
            nums[idx], nums[i] = nums[i], nums[idx]
            # use next integers to complete the permutations
            backtrack(idx + 1)
            nums[idx], nums[i] = nums[i], nums[idx] # backtrack
    backtrack(0)
    return list(visited)


# LC77. Combinations
from itertools import combinations
def combine(self, n, k):
    return list(combinations(range(1, n+1), k))

def combine(self, n, k):
    combs = [[]]
    for _ in range(k):
        combs = [[i] + c for c in combs for i in range(1, c[0] if c else n+1)]
    return combs
# for 4, 3: [[1],[2],[3],[4]],  [[1,2],[1,3],[2,3],[1,4],[2,4],[3,4]]
# and  [[1,2,3],[1,2,4],[1,3,4],[2,3,4]]



# LC1053. Previous Permutation With One Swap
def prevPermOpt1(self, arr: List[int]) -> List[int]:
    n = len(arr)
    idx = n-2 # looking for the first peak index inside from left
    while idx >= 0 and arr[idx] <= arr[idx+1]: idx -= 1
    if idx >= 0: # Otherwise, we have increasing series, just return
        midx = idx + 1  # now find max < arr[idx], swap with that
        for i in range(midx+1, n): # max != arr[idx], otherwise no change in swap
            if arr[idx] > arr[i] > arr[midx]: midx = i
        arr[idx], arr[midx] = arr[midx], arr[idx]
    return arr



# LC78. Subsets
def subsets(self, nums: List[int]) -> List[List[int]]:  # time and space O(2^N)
    if not nums: return []
    ret = [[]]
    for num in nums: ret += [ ss + [num] for ss in ret] # add one digit at a time
    return ret



# LC118. Pascal's Triangle
def generate(self, numRows):
    row, res = [1], []
    for n in range(numRows):
        res.append(row)
        row=[1] + [row[i] + row[i+1] for i in range(n)] + [1]
    return res


