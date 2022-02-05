
# LC75. Sort Colors   Dutch National Flag
def sortColors(self, nums: List[int]) -> None:
    if not nums: return
    lnth = len(nums)
    # https://en.wikipedia.org/wiki/Dutch_national_flag_problem
    lidx, hidx = 0, lnth - 1
    mid_h = 0
    color = 1 # among 0, 1, 2 different colors
    while mid_h <= hidx:  #O(n)
        if nums[mid_h] < color:
            nums[lidx], nums[mid_h] = nums[mid_h], nums[lidx]
            lidx += 1
            mid_h += 1
        elif nums[mid_h] > color:
            nums[mid_h], nums[hidx] = nums[hidx], nums[mid_h]
            hidx -= 1
        else: mid_h += 1

# LC2089. Find Target Indices After Sorting Array
def targetIndices(self, nums: List[int], target: int) -> List[int]:
    lt_count, eq_count = 0, 0  # partition. O(n), rathern than O(nlogn)
    for n in nums:
        if n < target: lt_count += 1
        elif n == target: eq_count += 1
    return list(range(lt_count, lt_count + eq_count))





