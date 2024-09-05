# 1752. Check if Array Is Sorted and Rotated
def check(self, nums: List[int]) -> bool:
    count = 0
    n = len(nums)
    for i in range(n):  # O(n)
        if nums[i] > nums[(i+1) % n]: count += 1
    return count <= 1

# LC33. Search in Rotated Sorted Array - return index, distinct values   search rotated array
def search(self, nums: List[int], target: int) -> int:  # O(logn)
    start, end = 0, len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] == target: return mid
        elif nums[mid] >= nums[start]:
            if nums[start] <= target < nums[mid]: end = mid - 1
            else: start = mid + 1  # cover 2 cases, target <> mid
        else:
            if nums[mid] < target <= nums[end]: start = mid + 1
            else: end = mid - 1
    return -1

# LC81. Search in Rotated Sorted Array II - check if num is in array, dupe values
def search(self, nums: List[int], target: int) -> bool:  # O(n) worst case
    start, end = 0, len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] == target: return True
        elif nums[mid] > nums[start]:
            if nums[start] <= target < nums[mid]: end = mid - 1
            else: start = mid + 1  # cover 2 cases, target <> mid
        elif nums[mid] < nums[start]:
            if nums[mid] < target <= nums[end]: start = mid + 1
            else: end = mid - 1
        else:  # equal and not target, could use a while. This is O(n)
            start += 1  # can't use mid, since it could jump out of range due to rotation, 1,0,1,1,1 with target 0
    return False

# LC153. Find Minimum in Rotated Sorted Array - unique elements  min rotate
def findMin(self, nums: List[int]) -> int:  # O(log)
    start, end = 0, len(nums) - 1
    while start < end:
        mid = (start + end) // 2
        if nums[mid] < nums[end]: end = mid
        else: start = mid + 1
    return nums[start]

# LC154. Find Minimum in Rotated Sorted Array II - contains dupes
def findMin(self, nums: List[int]) -> int:  # O(n)
    low, high = 0, len(nums)-1
    while high > low:
        pivot = low + (high - low) // 2
        if nums[pivot] < nums[high]: high = pivot  # we want eliminate higher values
        elif nums[pivot] > nums[high]: low = pivot + 1  # pivot is on left higher values
        else: high -= 1  # have to go slowly since min could be between
                         # the 'low' and 'high' index converge to the inflection point.
    return nums[low]
