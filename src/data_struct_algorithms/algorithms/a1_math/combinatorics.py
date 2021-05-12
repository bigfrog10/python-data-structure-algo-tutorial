# LC31. Next Permutation
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
# LC46. Permutations
def permute(self, nums: List[int]) -> List[List[int]]:
    n, output = len(nums), []
    def swap(first = 0):
        if first == n: output.append(nums[:]) # if all integers are swapped
        for i in range(first, n):
            # place i-th integer first in the current permutation
            nums[first], nums[i] = nums[i], nums[first]
            # use next integers to complete the permutations
            swap(first + 1)
            # backtrack
            nums[first], nums[i] = nums[i], nums[first]
    swap(0)
    return output
def permute(self, nums: List[int]) -> List[List[int]]:
    ans = [[nums[0]]]
    for num in nums[1:]:
        new_arr = []
        for each in ans:
            for i in range(len(each)+1):
                new_arr.append(each[:i] + [num] + each[i:])
        ans = new_arr
    return ans
# LC47. Permutations II
def permuteUnique(self, nums): # best
    ans = [[nums[0]]]
    for n in nums[1:]:
        new_ans = []
        for p in ans:
            for i in range(len(p)+1):
                new_ans.append(p[:i] + [n] + p[i:])
                # we insert dups only from left side.
                if i < len(p) and p[i] == n: break #handles duplication
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
# LC78. Subsets
def subsets(self, nums: List[int]) -> List[List[int]]:  # time and space O(2^N)
    if not nums: return []
    ret = [[]]
    for num in nums: ret += [ ss + [num] for ss in ret] # add one digit at a time
    return ret
# LC282. Expression Add Operators
def addOperators(self, num: str, target: int) -> List[str]:
    n = len(num)
    res = []
    def dfs(idx, expr, cur, last): # cur is the current value, last is last term
        if idx == n:
            if cur == target: res.append(expr)
            return
        for i in range(idx + 1, n + 1): # n+1 because we have num[idx:i]
            #if i == idx + 1 or (i > idx + 1 and num[idx] != "0"): # prevent "05"
            s, x = num[idx:i], int(num[idx:i])  # s could '0'
            if last == None: dfs(i, s, x, x)
            else:
                dfs(i, expr+"+"+s, cur + x, x)
                dfs(i, expr+"-"+s, cur - x, -x)
                # This is to handle 1 + 2 * 3, we need to backout 2 and add 2 * 3.
                dfs(i, expr+"*"+s, cur-last+last*x, last*x)
            if num[idx] == '0': break # after idx+1 we break out otherwise we have 01
    dfs(0, '', 0, None)
    return res
# LC241. Different Ways to Add Parentheses
def diffWaysToCompute(self, input: str) -> List[int]:
    listFinal = []
    if '+' not in input and '-' not in input and '*' not in input:
        listFinal.append(int(input)) # base case
    for i, v in enumerate(input):
        if v == '+' or v == '-' or v == '*':  # break by operators, then recursion down
            listFirst = self.diffWaysToCompute(input[0: i])
            listSecond = self.diffWaysToCompute(input[i + 1:])
            for i, valuei in enumerate(listFirst): # now combine results
                for j, valuej in enumerate(listSecond):
                    if v == '+': listFinal.append(valuei + valuej)
                    elif v == '-': listFinal.append(valuei - valuej)
                    else: listFinal.append(valuei * valuej)

    return listFinal
# LC17. Letter Combinations of a Phone Number, top100
def letterCombinations(self, digits):
    dict = {'2':"abc", '3':"def",  '4':"ghi", '5':"jkl",
            '6':"mno", '7':"pqrs", '8':"tuv", '9':"wxyz"}
    cmb = [''] if digits else []
    for d in digits: cmb = [p + q for p in cmb for q in dict[d]]
    return cmb
# LC118. Pascal's Triangle
def generate(self, numRows):
    row, res = [1], []
    for n in range(numRows):
        res.append(row)
        row=[1] + [row[i] + row[i+1] for i in range(n)] + [1]
    return res
# LC279. Perfect Squares, top100. minimal -> BFS
def numSquares(self, n):
    square_nums = [i * i for i in range(1, int(n**0.5)+1)] # list of square numbers that are less than `n`
    queue, level = {n}, 0
    while queue: # BFS
        level += 1
        next_queue = set()
        for remainder in queue: # construct the queue for the next level
            for square_num in square_nums:
                if remainder == square_num: return level  # find the node!
                elif remainder < square_num: break # overed, no need to go further, cut branches
                else: next_queue.add(remainder - square_num)
        queue = next_queue
    return level


