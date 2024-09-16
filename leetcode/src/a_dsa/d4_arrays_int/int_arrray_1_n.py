
# LC1920. Build Array from Permutation - array permute
# https://leetcode.com/problems/build-array-from-permutation/discuss/1315926/Python-O(n)-Time-O(1)-Space-w-Full-Explanation
def buildArray(self, nums: List[int]) -> List[int]:  # O(1) space
    q = len(nums)
    for i, c in enumerate(nums):
        nums[i] += q * (nums[c] % q)
    for i,_ in enumerate(nums):
        nums[i] //= q
    return nums

# LC442. Find All Duplicates in an Array  dupes in array
def findDuplicates(self, nums: List[int]) -> List[int]:  # run it again to restore
    ans = []
    for num in nums:
        if nums[abs(num) - 1 ] < 0: # Already Marked --> Repeated Guy
            ans.append(abs(num))
        else: # Not Marked
            nums[ abs(num) - 1 ] = - (nums[ abs(num) - 1 ])
    return ans

# LC268. Missing Number - only one in [0, n]
def missingNumber(self, nums: List[int]) -> int:
    s = sum(nums)
    n = len(nums)
    t = n * (n + 1) // 2
    return t - s
def missingNumber(self, nums):
    missing = len(nums)
    for i, num in enumerate(nums):
        missing ^= i ^ num
    return missing

# LC448. Find All Numbers Disappeared in an Array - missing in [1, n]
def findDisappearedNumbers(self, nums: List[int]) -> List[int]:  # O(n)
    for i in range(len(nums)):
        index = abs(nums[i]) - 1
        nums[index] = - abs(nums[index])
    return [i + 1 for i in range(len(nums)) if nums[i] > 0]
def findDisappearedNumbers(self, nums: List[int]) -> List[int]:  # to restore originals
    n1 = nums[:]
    for i in range(len(nums)):
        index = abs(nums[i]) - 1
        nums[index] = - abs(nums[index])
    res = [i + 1 for i in range(len(nums)) if nums[i] > 0]
    for i in range(len(nums)):
        index = abs(nums[i]) - 1
        nums[index] = abs(nums[index])
    for i in range(len(n1)):
        if n1[i] != nums[i]:
            raise Exception(f'{n1[i]} {nums[i]}')
    return res

# LC287. Find the Duplicate Number - only 1 dupe, could be >1 copy   dupe num >1
def findDuplicate(self, nums: List[int]) -> int:  # O(1) space and O(n) time
    for num in nums:  # repeating number could repeat many times
        if nums[abs(num)] < 0:  # 1, 3, 4, 2, 2 -> 1, -3, -4, -2, -2
            ans = abs(num)
            break
        nums[abs(num)] = -nums[abs(num)]
    for i in range(len(nums)):  # restore nums
        nums[i] = abs(nums[i])
    return ans
def findDuplicate(self, nums):  # O(1) space and O(n) time, Floyd's Tortoise and Hare (Cycle Detection)
    tortoise = hare = nums[0]
    while True:
        tortoise = nums[tortoise]
        hare = nums[nums[hare]]
        if tortoise == hare: break
    tortoise = nums[0]
    while tortoise != hare:  # Find the "entrance" to the cycle.
        tortoise = nums[tortoise]
        hare = nums[hare]
    return hare
