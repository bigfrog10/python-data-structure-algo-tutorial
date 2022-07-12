
# LC1151. Minimum Swaps to Group All 1's Together
def minSwaps(self, data: List[int]) -> int:  # O(n) and O(1)
    ones = sum(data)
    cnt_one = max_one = 0
    left = right = 0
    while right < len(data):
        # updating the number of 1's by adding the new element
        cnt_one += data[right]
        right += 1
        # maintain the length of the window to ones
        if right - left > ones:
            # updating the number of 1's by removing the oldest element
            cnt_one -= data[left]
            left += 1
        # record the maximum number of 1's in the window
        max_one = max(max_one, cnt_one)
    return ones - max_one

# LC525. Contiguous Array - longest subarray with equal # of 1 and 0, 01 array
def findMaxLength(self, nums: List[int]) -> int:  # O(n) time and space
    c2i = {} # store value to index, cache
    maxlen = count = 0
    for i, n in enumerate(nums):
        count += 1 if n else -1
        if count == 0: maxlen = max(maxlen, i+1)  # balanced from 0 to i
        if count in c2i: maxlen = max(maxlen, i - c2i[count])
        else: c2i[count] = i  # do this on first occurrence for longest cases
    return maxlen
