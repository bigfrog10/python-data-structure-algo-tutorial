
# LC948. Bag of Tokens score token score
def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
    score = 0
    tokens.sort()  # O(nlogn) time
    high, low = len(tokens) - 1, 0
    while low <= high:
        # When we have enough power, play lowest token face-up
        if power >= tokens[low]:
            score += 1
            power -= tokens[low]
            low += 1
        # We don't have enough power to play a token face-up
        # If there is at least one token remaining,
        # and we have enough score, play highest token face-down
        elif low < high and score > 0:
            score -= 1
            power += tokens[high]
            high -= 1
        else: return score
    return score

# LC2824. Count Pairs Whose Sum is Less than Target  pair < target pairs < target
def countPairs(self, nums: List[int], target: int) -> int:
    ns = sorted(nums)  # O(nlogn)
    count = left = 0
    right = len(nums) - 1
    while left < right:
        if ns[left] + ns[right] < target:
            count += right - left
            left += 1
        else: right -= 1
    return count

# LC912. Sort an Array  merge sort array
def sortArray(self, nums: List[int]) -> List[int]:  # O(nlogn) time, O(n) space
    def merge(left, right):
        res = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1
        res.extend(left[i:])  # remaining, only one of these used
        res.extend(right[j:])
        return res
    n = len(nums)
    if n == 1: return nums
    list1 = self.sortArray(nums[:n//2])
    list2 = self.sortArray(nums[n//2:])
    return merge(list1, list2)

def sortArray(self, N: List[int]) -> List[int]:  # quick sort
    def quicksort(A, I, J):
        if J - I <= 1: return
        p = partition(A, I, J)
        quicksort(A, I, p), quicksort(A, p + 1, J)
    def partition(A, I, J):
        A[J-1], A[(I + J - 1)//2], i = A[(I + J - 1)//2], A[J-1], I
        for j in range(I,J):
            if A[j] < A[J-1]: A[i], A[j], i = A[j], A[i], i + 1
        A[J-1], A[i] = A[i], A[J-1]
        return i
    quicksort(N,0,len(N))
    return N
def sortArray(self, nums: List[int]) -> List[int]:  # bucket sort, O(n)
    bucket = [0]*100001
    for x in nums: bucket[x + 50000] += 1
    ans = []
    for i, x in enumerate(bucket, -50000):
        ans.extend([i]*x)
    return ans