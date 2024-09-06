
# LC1567. Maximum Length of Subarray With Positive Product
def getMaxLen(self, nums: List[int]) -> int:  # O(n) and O(1)
    ans = pos = neg = 0  # length of pos or neg product accumulatively
    for x in nums:
        if x > 0:
            pos += 1
            neg = neg + 1 if neg else 0
        elif x < 0:  # do not break this assignment
            pos, neg = 1 + neg if neg else 0, 1 + pos
        else: pos = neg = 0 # reset
        ans = max(ans, pos)
    return ans
