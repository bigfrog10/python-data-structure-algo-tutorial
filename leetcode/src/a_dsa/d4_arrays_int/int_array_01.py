
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
