from typing import List
from collections import Counter
import math
import functools

# LC1762. Buildings With an Ocean View - increasing stack
def findBuildings(self, heights: List[int]) -> List[int]:  # O(n)
    res = [len(heights) - 1]  # last building always has a view
    for i in range(len(heights)-1)[::-1]:
        if heights[i] > heights[res[-1]]: res.append(i)
    res.reverse()
    return res

# LC339. Nested List Weight Sum
def depthSum(self, nestedList: List[NestedInteger]) -> int:
    def dfs(nested_list, depth):
        total = 0
        for nested in nested_list:
            if nested.isInteger(): total += nested.getInteger() * depth
            else: total += dfs(nested.getList(), depth + 1)
        return total
    return dfs(nestedList, 1)

# LC364. Nested List Weight Sum II
def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
    # https://leetcode.com/problems/nested-list-weight-sum-ii/discuss/1027152/Python3-1-pass-bfs
    # good bfs idea
    queue = nestedList
    s1 = 0
    s2 = 0  # breathe sum
    q1 = []  # breathe layer
    while queue:
        e = queue.pop()
        if e.isInteger(): s2 += e.getInteger()
        else: q1.extend(e.getList())
        if not queue:
            s1 += s2 # keep add same value to simulate weight
            queue = q1
            q1 = []
    return s1

# LC238. Product of Array Except Self, top100
def productExceptSelf(self, nums: List[int]) -> List[int]:
    length = len(nums)
    ret = [0] * length
    ret[0] = 1
    for i in range(1, length): ret[i] = nums[i - 1] * ret[i - 1]
    tmp = 1
    for i in reversed(range(length)):
        ret[i] = ret[i] * tmp
        tmp *= nums[i]
    return ret

# LC983. Minimum Cost For Tickets
def mincostTickets(self, days: List[int], costs: List[int]) -> int:
    durations = [1, 7, 30]  # this should be in parameter passed in
    day_set = set(days)
    @lru_cache(None)
    def dp(i):  # BFS on days
        if i > days[-1]: return 0
        elif i in day_set: # if we do travel today
            return min(dp(i + d) + c for c, d in zip(costs, durations))
        else: return dp(i + 1) # wait for next day if we don't travel today
    # print(dp.cache_info())
    return dp(days[0])

# LC1868. Product of Two Run-Length Encoded Arrays
def findRLEArray(self, encoded1: List[List[int]], encoded2: List[List[int]]) -> List[List[int]]:
    res, l, r = [], 0, 0  # O(n + m), counts of unique numbers
    while encoded1[-1][-1] != 0:
        prod = encoded1[l][0] * encoded2[r][0]
        low = min(encoded1[l][1], encoded2[r][1])
        if res and res[-1][0] == prod: res[-1][1] += low # extend freq if same value
        else: res.append([prod, low])

        encoded1[l][1] -= low  # minus the finished range
        encoded2[r][1] -= low
        if encoded1[l][1] == 0: l += 1
        if encoded2[r][1] == 0: r += 1

# LC932. Beautiful Array
def beautifulArray(self, n: int) -> List[int]:
    res = [1]
    while len(res) < n:
        res = [i * 2 - 1 for i in res] + [i * 2 for i in res]
    return [i for i in res if i <= n]

# LC1460. Make Two Arrays Equal by Reversing Sub-arrays
def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
    return collections.Counter(target) == collections.Counter(arr)

# LC605. Can Place Flowers
def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    if not flowerbed and n > 0: return False
    m = len(flowerbed)
    idx = count = 0
    while idx < m:
        if flowerbed[idx] == 0:
            if (idx == 0 or flowerbed[idx-1] == 0) and (idx == m -1 or flowerbed[idx+1] == 0):
                flowerbed[idx] = 1
                count += 1
        if count >= n: return True
        idx += 2 if idx + 2 < m else 1
    return False

# LC41. First Missing Positive, top100
def firstMissingPositive(self, nums: List[int]) -> int:
    # missing is in [1 ..., len(nums)] and we care only positives
    positives = set(x for x in nums if 0 < x <= len(nums))
    n = len(positives)
    if n == 0: return 1  # if all nums are 0, then next is 1 and 1 is missing
    for i in range(1, n+1):  # this order honors smallest missing
        if i not in positives: return i
    return n + 1
def firstMissingPositive(self, nums: List[int]) -> int:
    n = len(nums)
    for i in range(n):  # clean up so range is [1, n]
        if nums[i] < 1 or nums[i] > n: nums[i] = 0
    for i in range(n):
        if nums[i] > 0:  # 1 <= nums[i] % (n + 1) <= n:
            ind = nums[i] % (n + 1) - 1  # ensure ind in bound.
            nums[ind] += n + 1  # seen == a > n
    for i in range(n):
        if nums[i] <= n: return i + 1  # not seen == a <= n
    return n + 1

# LC219. Contains Duplicate II
def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    if k == 0: return False
    showed = set()  # if we have 2 same elements with k frame, then we are done.
    for idx, v in enumerate(nums):
        if v in showed: return True  # if we have 2 same elements, then we are done
        if len(showed) == k: showed.remove(nums[idx - k])
        showed.add(v)
    return False

# LC274. H-Index
def hIndex(self, citations: List[int]) -> int:  # O(n), better than sorting O(nlogn)
    n = len(citations)  # Counting without sorting
    papers = [0] * (n+1)
    for c in citations: papers[min(n, c)] += 1  # how many papers has this citation count
    sumc = 0
    for i in reversed(range(n+1)):
        sumc += papers[i]  # accumulate from end
        if sumc >= i: return i  # has at least i citations for all right side together
    return 0

# LC275. H-Index II
def hIndex(self, citations: List[int]) -> int:  # O(logn)
    n = len(citations)
    left, right = 0, n  # n-1 does not work for 1 element array
    while left < right:  # sorted in asc, so we use n - i for larger citations
        mid = left + (right - left) // 2
        if citations[mid] < n - mid: left = mid + 1
        else: right = mid
    return n - left

# LC1547. Minimum Cost to Cut a Stick
def minCost(self, n: int, cuts: List[int]) -> int:  # O(n^3)
    cc = sorted(cuts)
    @cache
    def cost(left, right):  # O(n^2)
        mincost = float('inf')
        idx1, idx2 = bisect.bisect_right(cc, left), bisect.bisect_left(cc, right)
        for i in cc[idx1:idx2]:  # O(n)
            new_cut_cost = cost(left, i) + cost(i, right)
            # right - left is the current cost
            mincost = min(mincost, new_cut_cost + right - left)
        mincost = 0 if mincost == float('inf') else mincost
        return mincost
    res = cost(0, n)
    return res

# LC2025. Maximum Number of Ways to Partition an Array
def waysToPartition(self, nums: List[int], k: int) -> int:
    # https://leetcode.com/problems/maximum-number-of-ways-to-partition-an-array/discuss/1499026/Short-Python-solution-Compute-prefix-sums%3A-O(n)
    prefix_sums = list(accumulate(nums))
    total_sum = prefix_sums[-1]
    # not replace with k, :-1 is because it's half sum, -1 is the other half, [0, 0, 0]
    best = prefix_sums[:-1].count(total_sum // 2) if total_sum % 2 == 0 else 0
    # diff = after pivot - before pivot = total sum - prefix_sum * 2, exclude last
    # [0, 1, 0], 0 requires last exclusion
    after_counts = Counter(total_sum - 2 * prefix_sum for prefix_sum in prefix_sums[:-1])
    before_counts = Counter()
    best = max(best, after_counts[k - nums[0]])  # If we change first num
    for prefix, x in zip(prefix_sums, nums[1:]):  # O(n)
        gap = total_sum - 2 * prefix  # diff need to fix
        after_counts[gap] -= 1
        before_counts[gap] += 1
        # k-num[i] is the diff to replace num[i], and the diff of presums
        # This value, for a fixed i, is the count of indices j with j > i that satisfy gap[j] == k - nums[i],
        # plus the number of indices j with 1 <= j <= i such that -gap[j] == k - nums[i]
        best = max(best, after_counts[k - x] + before_counts[x - k])
    return best

# LC532. K-diff Pairs in an Array
def findPairs(self, nums: List[int], k: int) -> int:
    result = 0
    counter = Counter(nums)
    for x in counter:
        if k > 0 and x + k in counter: result += 1
        elif k == 0 and counter[x] > 1: result += 1
    return result

# LC974. Subarray Sums Divisible by K
def subarraysDivByK(self, A: List[int], K: int) -> int:
    if not A: return 0
    cumu = list(accumulate(A))
    # pre-append 0 because we count x // K == 0 in down below formula.
    res = [0] + [x % K for x in cumu]
    counts = Counter(res) # number of cumus having same residue.
    # once we subtract any of these 2, we have the subarray sum divided by K.
    # so selecting 2 elements has C(C-1) / 2 possibilities.
    return sum(c * (c - 1) // 2 for c in counts.values())

# LC765. Couples Holding Hands
def minSwapsCouples(self, row):
    idxs, swap = {}, 0
    idxs = {x:i for (i,x) in enumerate(row)}
    for i in range(0, len(row), 2):
        partner = row[i] ^ 1
        j = idxs[partner]
        if j-i != 1:
            row[i+1], row[j] = row[j], row[i+1]
            idxs[row[j]] = j
            swap += 1
    return swap

# LC268. Missing Number
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
# LC66. Plus One
def plusOne(self, digits: List[int]) -> List[int]:
    for i in reversed(range(len(digits))):
        if digits[i] == 9: digits[i] = 0
        else:
            digits[i] += 1
            return digits  # when we don't have carry
    return [1] + digits  # when we have carry

# LC1748. Sum of Unique Elements
def sumOfUnique(self, nums: List[int]) -> int:
    return sum(a for a, c in collections.Counter(nums).items() if c == 1)

# LC989. Add to Array-Form of Integer
def addToArrayForm(self, num: List[int], k: int) -> List[int]:
    for i in range(len(num) - 1, -1, -1):
        k, num[i] = divmod(num[i] + k, 10)  # treat k as carry
    return [int(i) for i in str(k)] + num if k else num

# LC1231. Divide Chocolate
def maximizeSweetness(self, A, K):
    left, right = 1, sum(A) // (K + 1)
    while left < right:
        mid = (left + right + 1) // 2
        cur = cuts = 0
        for a in A: ## doing cuts now
            cur += a
            if cur >= mid:
                cuts += 1
                cur = 0
        if cuts > K: left = mid
        else: right = mid - 1
    return right

# LC442. Find All Duplicates in an Array
def findDuplicates(self, nums: List[int]) -> List[int]:  # run it again to restore
    res = []
    for x in nums:
        if nums[abs(x)-1] < 0: res.append(abs(x))
        else: nums[abs(x)-1] *= -1
    return res

# LC349. Intersection of Two Arrays
def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    set1 = set(nums1)
    set2 = set(nums2)
    return list(set2 & set1)

# LC448. Find All Numbers Disappeared in an Array
def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
    for i in range(len(nums)):
        index = abs(nums[i]) - 1
        nums[index] = - abs(nums[index])
    return [i + 1 for i in range(len(nums)) if nums[i] > 0]

# LC189. Rotate Array
def rotate(self, nums: List[int], k: int) -> None:
    n = len(nums)
    k = k % n
    nums[k:], nums[:k] = nums[:n-k], nums[n-k:]

# LC287. Find the Duplicate Number  Floyd's Tortoise and Hare (Cycle Detection)
def findDuplicate(self, nums: List[int]) -> int:  # O(1) space and O(n) time
    for num in nums:  # repeating number could repeat many times
        if nums[abs(num)] < 0:
            ans = abs(num)
            break
        nums[abs(num)] = -nums[abs(num)]
    for i in range(len(nums)): # restore nums
        nums[i] = abs(nums[i])
    return ans

# LC169. Majority Element
def majorityElement(self, nums: List[int]) -> int:
    counts = collections.Counter(nums)
    return max(counts.keys(), key=counts.get)

# LC1295. Find Numbers with Even Number of Digits
def findNumbers(self, nums: List[int]) -> int:
    return sum(~len(str(x)) & 1 for x in nums)
def findNumbers1(self, nums: List[int]) -> int:
    return sum(int(math.log10(n)) % 2 for n in nums) # log10(n) + 1 is the # of digits.

# LC846. Hand of Straights
def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
    if len(hand) % groupSize != 0: return False
    c = Counter(hand)
    for i in sorted(c):  # O(nlogn), sort by key
        if c[i] == 0: continue
        cnt = c[i]
        for j in range(groupSize):
            c[i + j] -= cnt
            if c[i + j] < 0: return False
    return True

# LC228. Summary Ranges
def summaryRanges(self, nums: List[int]) -> List[str]:
    pointer, ans = 0, []
    for i, n in enumerate(nums):
        if i == len(nums) - 1 or nums[i+1] - nums[i] > 1:
            ans.append(str(nums[pointer]) + '->' + str(n) if nums[pointer] != n else str(n))
            pointer = i+1
    return ans

# LC322. Coin Change
def coinChange(self, coins: List[int], amount: int) -> int:
    if amount == 0:  return 0 # 680 ms, it also is O(c^(amount / min(coins)))
    coins.sort(reverse=True) # we try to put larger coins to reduce numbers of coins
    queue, visited = deque([(0, 0)]), {0}
    while queue:
        for _ in range(len(queue)): # BFS
            amt, count = queue.popleft()
            for coin in coins:
                namt, nc = amt + coin, count + 1
                if namt == amount: return nc
                elif namt < amount and namt not in visited:
                    visited.add(namt)
                    queue.append((namt, nc))
    return -1

# LC518. Coin Change 2
def change(self, amount: int, coins: List[int]) -> int:
    if not coins: return 1 if amount == 0 else 0
    dp = [1] + [0] * amount
    for c in coins:
        for i in range(c, amount+1): dp[i] += dp[i-c]
    return dp[amount]
def change(self, amount: int, coins: List[int]) -> int:
    coins.sort(reverse=True)  # make it much faster, but not necessary
    @lru_cache(maxsize=None)
    def f(i, amount):  # having i is to remove double count, (2, 1, 1), (1, 2, 1), (1,1,2)
        if amount == 0 : return 1
        if amount < 0 or i >= len(coins): return 0
        return f(i, amount-coins[i]) + f(i+1, amount)
    return f(0, amount)

# LC628. Maximum Product of Three Numbers
def maximumProduct(self, nums: List[int]) -> int:
    max1 = max2 = max3 = float('-inf')  # 1 > 2 > 3
    min1 = min2 = float('inf')  # 1 < 2, to deal with negative numbers
    for n in nums:  # O(n)
        if n > max1: max1, max2, max3 = n, max1, max2
        elif n > max2: max2, max3 = n, max2
        elif n > max3: max3 = n

        if n < min1: min1, min2 = n, min1
        elif n < min2: min2 = n
    return max(max1 * max2 * max3, max1 * min1 * min2)

# LC1299. Replace Elements with Greatest Element on Right Side
def replaceElements(self, arr: List[int]) -> List[int]:
    mx = -1
    for i in range(len(arr))[::-1]:
        arr[i], mx = mx, max(mx, arr[i])
    return arr

# LC414. Third Maximum Number
def thirdMax(self, nums: List[int]) -> int:
    v = [float('-inf'), float('-inf'), float('-inf')]
    for num in nums:
        if num not in v:
            if num > v[0]:   v = [num, v[0], v[1]]
            elif num > v[1]: v = [v[0], num, v[1]]
            elif num > v[2]: v = [v[0], v[1], num]
    return max(nums) if float('-inf') in v else v[2]

# LC136. Single Number
def singleNumber(self, nums: List[int]) -> int:
    res = 0
    for i in nums: res ^= i
    return res

# LC137. Single Number II
def singleNumber(self, nums: List[int]) -> int:
    seen_once = seen_twice = 0
    for num in nums:
        # first appearance:
        # add num to seen_once
        # don't add to seen_twice because of presence in seen_once

        # second appearance:
        # remove num from seen_once
        # add num to seen_twice

        # third appearance:
        # don't add to seen_once because of presence in seen_twice
        # remove num from seen_twice
        seen_once = ~seen_twice & (seen_once ^ num)
        seen_twice = ~seen_once & (seen_twice ^ num)
    return seen_once

# LC217. Contains Duplicate
def containsDuplicate(self, nums: List[int]) -> bool:
    showed = set()
    for i in nums:
        if i in showed: return True
        showed.add(i)
    return False
def containsDuplicate(self, nums):
    return len(set(nums)) < len(nums)

# LC485. Max Consecutive Ones
def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
    count = max_count = 0
    for num in nums:
        if num == 1: count += 1 # Increment the count of 1's by one.
        else:
            max_count = max(max_count, count)
            count = 0 # Reset count of 1.
    return max(max_count, count)

# LC945. Minimum Increment to Make Array Unique
def minIncrementForUnique(self, nums: List[int]) -> int: # O(n)
    if not nums: return 0
    count = collections.Counter(nums)
    maxv = max(nums)
    ans, dups = 0, []
    # we may have dups after singles, so use this range
    for x in range(maxv + len(nums)): # This is the largest range
        if count[x] >= 2: # save dups
            dups.extend([x] * (count[x] - 1))
        elif dups and count[x] == 0: # empty slot found
            ans += x - dups.pop() # get dup fill in this slot
    return ans

# LC179. Largest Number
def largestNumber(self, nums: List[int]) -> str:
    num = [str(x) for x in nums]
    # cmp(x, y) can be replaced by (x > y) - (x < y)
    # in this problem x is (a+b) and y is (b+a)
    cmp = lambda b, a: ((a+b)>(b+a))-((a+b)<(b+a))
    num.sort(key = functools.cmp_to_key(cmp))
    return ''.join(num).lstrip('0') or '0'










# LC775. Global and Local Inversions
def isIdealPermutation(self, A):
    return all(abs(i-x) <= 1 for i,x in enumerate(A))

# LC679. 24 Game
def judgePoint24(self, nums: List[int]) -> bool:
    if len(nums) == 1: return math.isclose(nums[0], 24)
    return any(self.judgePoint24([x] + rest)
               for a, b, *rest in itertools.permutations(nums)
               for x in {a+b, a-b, a*b, b and a/b})

# LC1583. Count Unhappy Friends
def unhappyFriends(self, n: int, preferences: List[List[int]], pairs: List[List[int]]) -> int:
    d, p = {}, preferences
    for x, y in pairs:
        d[x] = p[x][:p[x].index(y)] # prefer persons before y
        d[y] = p[y][:p[y].index(x)]
    res = 0
    for x in d:
        for u in d[x]:
            if x in d[u]: # x is in prefer list of u before his pair.
                res += 1
                break
    return res

# LC611. Valid Triangle Number
def triangleNumber(self, nums: List[int]) -> int:
    n = len(nums)
    nums.sort()
    count = 0
    for i in range(n-1,1,-1):
        hi, lo = i - 1, 0
        while lo < hi:
            if nums[hi]+nums[lo] > nums[i]:
                count += hi - lo
                hi -= 1
            else: lo += 1
    return count










# LC315. Count of Smaller Numbers After Self  # BBG hard
def countSmaller(self, nums: List[int]) -> List[int]:
    sorted_arr = [] # O(nlogn)
    rst = []
    for num in nums[::-1]:
        idx = bisect_left(sorted_arr, num)
        rst.append(idx)
        sorted_arr.insert(idx, num)  # this is o(n)
    return rst[::-1]

# LC493. Reverse Pairs
def reversePairs(self, nums: List[int]) -> int:
    import sortedcontainers
    brr = sortedcontainers.SortedList(nums)
    count = 0
    # anything smaller before larger is discarded.
    for i in range(len(nums)): # O(nlogn), loop is n, logn inside
        brr.discard(nums[i])
        k = brr.bisect_left((nums[i]+1)//2)
        count += k
    return count

# LC1470. Shuffle the Array
def shuffle(self, nums: List[int], n: int) -> List[int]:
    res = [0] * (2*n)
    res[::2] = nums[:n]
    res[1::2] = nums[n:]
    return res
def shuffle1(self, nums: List[int], n: int) -> List[int]:
    return [num for t in zip(nums[:n],nums[n:]) for num in t]

# LC412. Fizz Buzz
def fizzBuzz(self, n: int) -> List[str]:
    res = []
    for i in range(1, n+1):
        if i % 3 == 0 and i % 5 == 0: res.append('FizzBuzz')
        elif i % 3 == 0: res.append('Fizz')
        elif i % 5 == 0: res.append('Buzz')
        else: res.append(str(i))
    return res

# LC697. Degree of an Array
def findShortestSubArray(self, nums: List[int]) -> int:
    dt = collections.defaultdict(list)
    for i, v in enumerate(nums): dt[v].append(i)
    degree = max(len(v) for v in dt.values())
    return min(dt[k][-1] - dt[k][0] + 1 for k in dt if len(dt[k]) == degree)





# LC350. Intersection of Two Arrays II
def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
    counts1 = collections.Counter(nums1)
    counts2 = collections.Counter(nums2)
    res = [] # or we may just loop nums2 and manipulate counts.
    for i, v in counts1.items():
        c = min(v, counts2[i])
        for k in range(c): res.append(i)
    return res

# LC1578. Minimum Deletion Cost to Avoid Repeating Letters
def minCost(self, s: str, cost: List[int]) -> int:
    res = max_cost = 0 # O(n)
    for i in range(len(s)): # For each group of continuous same characters,
        if i > 0 and s[i] != s[i - 1]: # min cost = sum_cost(group) - max_cost(group)
            max_cost = 0 # preset this to prevent max added to res
        res += min(max_cost, cost[i]) # add min first, then update max next line
        max_cost = max(max_cost, cost[i]) # this ensures max is not added
    return res




