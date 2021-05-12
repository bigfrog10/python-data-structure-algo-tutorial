# LC209. Minimum Size Subarray Sum
def minSubArrayLen(self, s: int, nums: List[int]) -> int:
    total = left = 0
    result = len(nums) + 1
    for right, n in enumerate(nums):
        total += n
        while total >= s:
            result = min(result, right - left + 1)
            total -= nums[left]
            left += 1
    return result if result <= len(nums) else 0

# LC992. Subarrays with K Different Integers
def subarraysWithKDistinct(self, A: List[int], K: int) -> int:
    def atMostK(A, K): # we demand K diff ints
        count = collections.Counter()
        res = i = 0
        for j in range(len(A)): # move right
            if count[A[j]] == 0: K -= 1 # we don't have this char anymore
            count[A[j]] += 1
            while K < 0:
                count[A[i]] -= 1
                if count[A[i]] == 0: K += 1 # we need 1 more
                i += 1  # left pointer move right
            res += j - i + 1 # when k >= 0 # all substring starting j
        return res
    return atMostK(A, K) - atMostK(A, K - 1)

# LC424. Longest Repeating Character Replacement
def characterReplacement(self, s: str, k: int) -> int:
    char_freq = Counter()
    start = walker = 0  # 2 pointers
    max_freq = max_len = 0  # we keep track of these 2
    while walker < len(s):
        c = s[walker]
        char_freq[ s[walker] ] += 1
        max_freq = max(max_freq, char_freq[c]) # gauge for left pointer
        # within max_freq + k, we can replace k chars to make it same chars
        # if go beyond, we have to make it smaller.
        if walker - start + 1 > max_freq + k:
            char_freq[ s[start] ] -= 1
            start += 1
        max_len = walker - start + 1  # this is monotonously increasing
        walker += 1
    return max_len

# LC239. Sliding Window Maximum, top100
def maxSlidingWindow(self, nums: 'List[int]', k: 'int') -> 'List[int]':
    n = len(nums)
    if n == 0 or k == 0: return []
    if k == 1: return nums
    deq = deque()
    def clean_deque(i):
        if deq and deq[0] == i - k: deq.popleft() # remove indexes of elements not from sliding window
        # remove from deq indexes of all elements
        # which are smaller than current element nums[i]
        while deq and nums[i] > nums[deq[-1]]: deq.pop() # decreasing queue
    output =[]
    for i in range(n):
        clean_deque(i)
        deq.append(i)
        if i >= k-1: output.append(nums[deq[0]])
    return output

# LC480. Sliding Window Median
def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
    window = sorted(nums[:k])
    medians = []
    for a, b in zip(nums, nums[k:] + [0]):
        medians.append((window[k//2] + window[~(k//2)]) / 2.)
        window.remove(a)
        bisect.insort(window, b)
    return medians
