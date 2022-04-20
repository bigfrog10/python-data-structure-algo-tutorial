
# LC75. Sort Colors   Dutch National Flag
def sortColors(self, nums: List[int]) -> None:
    p0 = 0  # for all idx < p0 : nums[idx < p0] = 0
    curr = 0  # curr is an index of element under consideration
    p2 = len(nums) - 1  # for all idx > p2 : nums[idx > p2] = 2
    while curr <= p2:
        if nums[curr] == 0:
            nums[p0], nums[curr] = nums[curr], nums[p0]
            p0 += 1
            curr += 1
        elif nums[curr] == 2:
            nums[curr], nums[p2] = nums[p2], nums[curr]
            p2 -= 1
        else: curr += 1

# LC2089. Find Target Indices After Sorting Array - target index
def targetIndices(self, nums: List[int], target: int) -> List[int]:
    lt_count, eq_count = 0, 0  # partition. O(n), rathern than O(nlogn)
    for n in nums:
        if n < target: lt_count += 1
        elif n == target: eq_count += 1
    return list(range(lt_count, lt_count + eq_count))





