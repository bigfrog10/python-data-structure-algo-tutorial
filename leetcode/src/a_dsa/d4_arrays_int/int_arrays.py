from typing import List
from collections import Counter
import math
import functools

# LC801 Minimum Swaps To Make Sequences Increasing    min swap 2 arrays increasing
def minSwap(self, A: List[int], B: List[int]) -> int:
    ans = sm = lg = mx = 0
    for x, y in zip(A, B):
        # need to reduce
        if mx < min(x, y): # prev max < current min
            ans += min(sm, lg) # treat so far subarray independent of the rest
            sm = lg = 0
        mx = max(x, y)
        if x < y: sm += 1 # count "x < y"  count both cases to see which one is smaller
        elif x > y: lg += 1 # count "x > y"
    return ans + min(sm, lg)
# https://leetcode.com/problems/minimum-swaps-to-make-sequences-increasing/solutions/932390/python3-two-counters/?envType=company&envId=facebook&favoriteSlug=facebook-three-months
# with if block, this is wrong [0,3,4,9,10]  [2,3,7,5,6], the two 3's are equal
# https://leetcode.com/problems/minimum-swaps-to-make-sequences-increasing/solutions/3835800/python-3-o-n-time-o-1-space-detailed-explanation/?envType=company&envId=facebook&favoriteSlug=facebook-three-months


# LC2970. Count the Number of Incremovable Subarrays I
def incremovableSubarrayCount(self, nums: List[int]) -> int: # O(n^3) time and O(1) space
    n, ans = len(nums), 0
    for i in range(n):
        for j in range(i, n):
            is_increasing = True
            lst = -1  # previous element
            for k in range(n):
                if i <= k <= j: continue  # exclude
                else:
                    is_increasing &= lst < nums[k]
                    lst = nums[k]
            ans += int(is_increasing)
    return ans

# LC2972. Count the Number of Incremovable Subarrays II  # O(nlogn)
def incremovableSubarrayCount(self, nums: List[int]) -> int:
    n = len(nums)  # O(nlogn) time, O(1) space
    res, i, j = 0, 0, n-1
    while i < n-1 and nums[i] < nums[i+1]: i += 1
    while 0 < j and nums[j-1] < nums[j]: j -= 1
    if j<=i: return (n+1)*n//2  # nums strictly increasing, return all subarrays
    # for diminishing left array (from index 0), find insert idx in right array
    # every number starts from the insert idx represents a solution: [0...i, k...n-1]
    # including the right empty array, [0...i, ]
    while i >= 0:
        k = bisect.bisect(nums, nums[i], lo=j)
        res += n-k+1
        i -= 1
    # when left array becomes empty, each number in right array also represents a solution
    # [, j...n-1] including [,], the empty left and empty right
    res += n-j+1
    return res


# LC1826. Faulty Sensor
def badSensor(self, sensor1: List[int], sensor2: List[int]) -> int:
    n = len(sensor1)
    for i in range(n - 1):
        if sensor1[i] != sensor2[i]:
            if sensor2[i] != sensor1[i + 1]: return 1
            elif sensor1[i] != sensor2[i + 1]: return 2
    return -1

# LC349. Intersection of Two Arrays - return unique elems  array intersect array
def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    set1 = set(nums1)
    set2 = set(nums2)
    return list(set2 & set1)

# LC275. H-Index II
def hIndex(self, citations: List[int]) -> int:  # O(logn)
    n = len(citations)
    left, right = 0, n  # n-1 does not work for 1 element array
    while left < right:  # sorted in asc, so we use n - i for larger citations
        mid = (right + left) // 2
        # n - mid papers have c[mid] >= n - mid citations
        if citations[mid] < n - mid: left = mid + 1
        else: right = mid
    return n - left

# LC2094. Finding 3-Digit Even Numbers - 3 digit even
def findEvenNumbers(self, digits: List[int]) -> List[int]:
    counts = collections.Counter(digits)
    res = []
    for i in range(100, 1000, 2):
        checker = defaultdict(int)
        d = i
        while d > 0:
            checker[d % 10] += 1
            d = d // 10
        if all(checker[x] <= counts[x] for x in checker.keys()):
            res.append(i)
    return res

# LC697. Degree of an Array - array degree, max freq
def findShortestSubArray(self, nums: List[int]) -> int:
    dt = collections.defaultdict(list)
    for i, v in enumerate(nums): dt[v].append(i)
    degree = max(len(v) for v in dt.values())
    return min(dt[k][-1] - dt[k][0] + 1 for k in dt if len(dt[k]) == degree)

# LC1207. Unique Number of Occurrences
def uniqueOccurrences(self, arr: List[int]) -> bool:
    c = collections.Counter(arr)
    return len(c) == len(set(c.values()))

# LC455. Assign Cookies awesome parent
def findContentChildren(self, g: List[int], s: List[int]) -> int:  # O(NlogN+MlogM)
    i, j, g, s = 0, 0, sorted(g), sorted(s)
    while i < len(g) and j < len(s):
        i += g[i] <= s[j]
        j += 1
    return i

# LC238. Product of Array Except Self, top100  prod array prod except
def productExceptSelf(self, nums: List[int]) -> List[int]:  # O(n) time and O(1) space
    length = len(nums)
    ret = [0] * length
    ret[0] = 1
    for i in range(1, length): ret[i] = nums[i - 1] * ret[i - 1]
    tmp = 1
    for i in reversed(range(length)):
        ret[i] = ret[i] * tmp
        tmp *= nums[i]
    return ret

# LC849. Maximize Distance to Closest Person - max distance between seats
def maxDistToClosest(self, seats: List[int]) -> int:
    res, last, n = 0, -1, len(seats)
    for i in range(n):
        if seats[i]:
            res = max(res, i if last < 0 else (i - last) // 2)
            last = i
    return max(res, n - last - 1)

# LC179. Largest Number - after re-arrange array ints, re-arrange elem, array re-arrange  x + y > y + x
def largestNumber(self, num):  # O(nlogn)
    num = [str(x) for x in num]
    cmp = lambda x, y: (x > y) - (x < y)  # standard comparator
    # if b + a > a + b, then we choose b + a
    custcmp = lambda a, b: cmp(b + a, a + b)  # specific to this problem
    num.sort(key = functools.cmp_to_key(custcmp))
    return ''.join(num).lstrip('0') or '0'

# LC904. Fruit Into Baskets - Find out the longest length of subarrays with at most 2 different numbers fruit tree
# https://leetcode.com/problems/fruit-into-baskets/discuss/170745/Problem%3A-Longest-Subarray-With-2-Elements
def totalFruit(self, fruits: List[int]) -> int:  # O(n) time, O(1) space
    res = left = 0
    counts = Counter()
    for right, ftype in enumerate(fruits):
        counts[ftype] += 1
        while len(counts.keys()) > 2:
            f = fruits[left]
            counts[f] -= 1
            if counts[f] == 0: del counts[f]
            left += 1
        res = max(res, sum(counts.values()))
    return res
def totalFruit(self, fruits: List[int]) -> int:  # O(n) time and O(1) space
    res = cur = count_b = a = b = 0
    for c in fruits:
        cur = cur + 1 if c in (a, b) else count_b + 1
        count_b = count_b + 1 if c == b else 1
        if b != c: a, b = b, c
        res = max(res, cur)
    return res

# LC755. Pour Water
def pourWater(self, heights: List[int], volume: int, k: int) -> List[int]:  # O(n + V)
    N = len(heights)
    j = k # ! important optimization
    for i in range(volume):  # O(V)
        while j > 0 and heights[j] >= heights[j-1]: # totally, j travels n
            j -= 1
        while j < N - 1 and heights[j] >= heights[j+1]:
            j += 1
        while j > k and heights[j]== heights[j-1]:
            j -= 1
        heights[j] += 1
    return heights


# LC532. K-diff Pairs in an Array - k diff pairs k diff
def findPairs(self, nums: List[int], k: int) -> int:  # O(n) time and space
    result = 0
    counter = Counter(nums)
    for x in counter:
        if k > 0 and x + k in counter: result += 1
        elif k == 0 and counter[x] > 1: result += 1
    return result

# LC220. Contains Duplicate III - bounded by index diff and value diff
def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:  # O(n) time and space
    n = len(nums)  # bucket sort
    d = {}  # d is buckets
    w = t + 1  # because the problem states at most t, so we have to use interval [0, t]
    for i in range(n):
        m = nums[i] // w
        if m in d: return True
        if m - 1 in d and abs(nums[i] - d[m - 1]) < w: return True
        if m + 1 in d and abs(nums[i] - d[m + 1]) < w: return True
        d[m] = nums[i]
        if i >= k: del d[nums[i - k] // w]  # O(min(n, k)) space
    return False

# LC229. Majority Element II      appear more than n/3
def majorityElement(self, nums: List[int]) -> List[int]:  # O(n) time and O(1) space
    ctr = collections.Counter()
    for n in nums:
        ctr[n] += 1
        if len(ctr) == 3:  # There can be at most two majority elements which are more than ⌊n/3⌋ times.
            ctr -= collections.Counter(set(ctr))
    return [n for n in ctr if nums.count(n) > len(nums)/3]  # [3, 2, 3], 2 is not

# LC322. Coin Change - least number of coins to sum up to target
def coinChange(self, coins: List[int], amount: int) -> int:
    if amount == 0: return 0  # O(amount * (coins)))
    coins.sort(reverse=True)  # we try to put larger coins to reduce numbers of coins
    queue, visited = deque([(0, 0)]), {0}
    while queue:  # BFS
        amt, count = queue.popleft()
        for coin in coins:
            namt, nc = amt + coin, count + 1
            if namt == amount: return nc
            elif namt < amount and namt not in visited:
                visited.add(namt)
                queue.append((namt, nc))
    return -1

# LC518. Coin Change 2 - return # of combinations
def change(self, amount: int, coins: List[int]) -> int:  # O(amount) space time O(amount * len(coins))
    dp = [1] + [0] * amount  # 1 is for 0 amount and no coins
    for c in coins:
        for i in range(c, amount+1): dp[i] += dp[i-c]
    return dp[amount]
def change1(self, amount: int, coins: List[int]) -> int:  # O(amount * len(coins)) knapsack problem
    coins.sort(reverse=True)  # make it much faster, but not necessary
    @lru_cache(maxsize=None)
    def f(i, amount):  # having i is to remove double count, (2, 1, 1), (1, 2, 1), (1,1,2)
        if amount == 0: return 1
        if amount < 0 or i >= len(coins): return 0
        # use this coin or move to next coin
        return f(i, amount-coins[i]) + f(i+1, amount)
    return f(0, amount)

# LC1431. Kids With the Greatest Number of Candies extracandies
def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
    gauge = max(candies) - extraCandies
    return [candy >= gauge for candy in candies]

# LC746. Min Cost Climbing Stairs climb stairs
def minCostClimbingStairs(self, cost: List[int]) -> int:  # O(n) time and O(1) space
    go1 = go2 = 0
    for i in range(2, len(cost) + 1):
        go1, go2 = min(go1 + cost[i - 1], go2 + cost[i - 2]), go1
    return go1

# LC926. Flip String to Monotone Increasing - 01
def minFlipsMonoIncr(self, s: str) -> int:  # O(n) and O(1)
    ones = flip = 0
    for ch in s:
        if ch == '1': ones += 1  # when 1's come, not need to flip
        else: flip += 1  # when 0's come, either flip this 0 to 1, plus old flips
        flip = min(flip, ones)  # or flip old ones to 0
    return flip

# LC696. Count Binary Substrings - with same 0s and 1s
def countBinarySubstrings(self, s: str) -> int:
    ans, prev, cur = 0, 0, 1 # prev and curr strike lengths
    for i in range(1, len(s)):
        if s[i-1] != s[i]:
            ans += min(prev, cur) # 01/10, 0011/1100, etc
            prev, cur = cur, 1
        else: cur += 1
    return ans + min(prev, cur)

# LC164. Maximum Gap - max diff sorted
def maximumGap(self, nums: List[int]) -> int:  # O(n)  Pigeonhole Principle
    lo, hi, n = min(nums), max(nums), len(nums)
    if n <= 2 or hi == lo: return hi - lo
    B = defaultdict(list)
    for num in nums:
        ind = n-2 if num == hi else (num - lo) * (n-1) // (hi-lo)
        B[ind].append(num)

    cands = [[min(B[i]), max(B[i])] for i in range(n-1) if B[i]]
    return max(y[0]-x[1] for x,y in zip(cands, cands[1:]))

# LC1762. Buildings With an Ocean View - increasing stack
def findBuildings(self, heights: List[int]) -> List[int]:  # O(n)
    res = [len(heights) - 1]  # last building always has a view
    for i in range(len(heights)-1)[::-1]:
        if heights[i] > heights[res[-1]]: res.append(i)
    res.reverse()
    return res

# LC339. Nested List Weight Sum - top elem has less weight , use dfs  nested sum
def depthSum(self, nestedList: List[NestedInteger]) -> int:  # O(all ints) time, O(D) space
    def dfs(nested_list, depth):
        total = 0
        for nested in nested_list:
            if nested.isInteger(): total += nested.getInteger() * depth
            else: total += dfs(nested.getList(), depth + 1)
        return total
    return dfs(nestedList, 1)

# LC364. Nested List Weight Sum II - top elem has more weight, accumu bfs  nested sum
def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
    queue = nestedList  ## O(n), n = total number of elements
    res, s2, q1 = 0, 0, []  # return, level sum, level queue
    while queue:  # BFS
        e = queue.pop()
        if e.isInteger(): s2 += e.getInteger()
        else: q1.extend(e.getList())
        if not queue:
            res += s2 # keep add same value to simulate weight
            queue, q1 = q1, []
    return res

# LC2149. Rearrange Array Elements by Sign  rearr sign rearr array arrange sign
def rearrangeArray(self, nums: List[int]) -> List[int]:
    n = len(nums)
    ans = [0] * n
    pos_index, neg_index = 0, 1
    for i in range(n):
        if nums[i] > 0:
            ans[pos_index] = nums[i]
            pos_index += 2
        else:
            ans[neg_index] = nums[i]
            neg_index += 2
    return ans

# LC2214. Minimum Health to Beat Game  min health game
def minimumHealth(self, damage: List[int], armor: int) -> int:
    return 1 + sum(damage) - min(max(damage), armor)

# LC932. Beautiful Array
# Given a beautiful array A: A*c, A + c, and delete elements from A are still beautiful.
# Given two Beautiful array A and B, whose elements are odd and even respectively, then concatenation array A + B
def beautifulArray(self, n: int) -> List[int]:
    res = [1]
    while len(res) < n:
        res = [i * 2 - 1 for i in res] + [i * 2 for i in res]
    return [i for i in res if i <= n]

# LC1460. Make Two Arrays Equal by Reversing Sub-arrays - reverse array  2 arrays equal  reverse array equal reverse subarray
def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
    return collections.Counter(target) == collections.Counter(arr)
# ignore positions

# LC605. Can Place Flowers
def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    for i in range(len(flowerbed)):  # O(len(s)) time, O(1) space
        left = i == 0 or flowerbed[i-1] == 0
        right = i == len(flowerbed) - 1 or flowerbed[i+1] == 0
        if left and right and flowerbed[i] == 0:
            flowerbed[i] = 1
            n -= 1
    return n <= 0

# LC41. First Missing Positive, top100   1st missing positive
def firstMissingPositive(self, nums: List[int]) -> int:  # O(n) time O(1) space
    # https://leetcode.com/problems/first-missing-positive/solutions/4926146/unveiling-the-ultimate-strategy-100-user-beating-solution/?envType=company&envId=apple&favoriteSlug=apple-six-months
    n = len(nums)
    b = False
    for i in range(n):
        if nums[i] == 1: b = True  # mark whether 1 shows up
        if nums[i] > n or nums[i] <= 0: nums[i] = 1  #
    if not b: return 1
    for i in range(n):   # change value [1, n] to negative, so -1 to [0, n-1]
        if nums[abs(nums[i]) - 1] > 0: nums[abs(nums[i]) - 1] *= -1
    for i in range(n):
        if nums[i] > 0: return i + 1
    return n + 1
# to restore original values, shift negatives by -2n, keep >= n as is

# LC219. Contains Duplicate II - duplicates within index range k distinct indices dupe k
def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:  # O(n) time and O(k) space
    if k == 0: return False
    showed = set()  # if we have 2 same elements with k frame, then we are done.
    for idx, v in enumerate(nums):
        if v in showed: return True  # if we have 2 same elements, then we are done
        if len(showed) == k: showed.remove(nums[idx - k])
        showed.add(v)
    return False
def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    if k == 0: return False
    cache = {}
    for idx, v in enumerate(nums):
        if v in cache and idx - cache[v] <= k: return True
        cache[v] = idx
    return False

# LC1262. Greatest Sum Divisible by Three
def maxSumDivThree(self, nums: List[int]) -> int:
    # https://leetcode.com/problems/greatest-sum-divisible-by-three/discuss/431077/JavaC%2B%2BPython-One-Pass-O(1)-space
    # dp[0] = largest sum which is divisible by 3
    # dp[1] = largest sum when divided by 3, remainder = 1
    # dp[2] = largest sum when divided by 3, remainder = 2
    # dp_cur[(rem + num) % 3] = max(dp_prev[(rem + num) % 3], dp_prev[rem]+num)
    dp = [0, 0, 0]
    for a in nums:
        for i in dp[:]:
            dp[(i + a) % 3] = max(dp[(i + a) % 3], i + a)
    return dp[0]

# LC274. H-Index
def hIndex(self, citations: List[int]) -> int:  # O(n) time and space, better than sorting O(nlogn)
    n = len(citations)  # Counting without sorting
    papers = [0] * (n+1)
    for c in citations: papers[min(n, c)] += 1  # how many papers has this citation count
    sumc = 0  # number of papers with at least i citations
    for i in reversed(range(n+1)):
        sumc += papers[i]  # accumulate from end, sumc is the number of papers having at least i citations.
        if sumc >= i: return i  # has at least i citations for all right side together
    return 0



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



# LC765. Couples Holding Hands
def minSwapsCouples(self, row):  # O(n)
    idxs = {x:i for (i,x) in enumerate(row)}
    swap = 0
    for i in range(0, len(row), 2):
        partner = row[i] ^ 1
        j = idxs[partner]
        if j-i != 1:  # not next to each other
            row[i+1], row[j] = row[j], row[i+1]
            idxs[row[j]] = j
            swap += 1
    return swap

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

# LC989. Add to Array-Form of Integer - array + integer   add int form  add int array form
def addToArrayForm(self, num: List[int], k: int) -> List[int]:
    for i in range(len(num) - 1, -1, -1):
        k, num[i] = divmod(num[i] + k, 10)  # treat k as carry
    return [int(i) for i in str(k)] + num if k else num  # else k=0

# LC189. Rotate Array
def rotate(self, nums: List[int], k: int) -> None:
    n = len(nums)
    k = k % n
    nums[k:], nums[:k] = nums[:n-k], nums[n-k:]

# LC169. Majority Element
def majorityElement(self, nums: List[int]) -> int:  # O(n) runtime and space
    counts = collections.Counter(nums)
    return max(counts.keys(), key=counts.get)
def majorityElement(self, nums):  # O(n) runtime, O(1) space, Boyer-Moore Voting Algorithm
    candidate, count = None, 0
    for num in nums:
        if count == 0: candidate = num  # discard prefix, pairs of 1 majority and not.
        count += 1 if num == candidate else -1
    return candidate

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
        if c[i] == 0: continue  # count is zero, so ignore
        cnt = c[i]
        for j in range(groupSize):
            c[i + j] -= cnt
            if c[i + j] < 0: return False
    return True

# LC228. Summary Ranges number range cover numbers cover range
def summaryRanges(self, nums: List[int]) -> List[str]:
    left, ans = 0, []
    for i, n in enumerate(nums):
        if i == len(nums) - 1 or nums[i+1] - nums[i] > 1:
            ans.append(str(nums[left]) + '->' + str(n) if nums[left] != n else str(n))
            left = i+1
    return ans

# LC628. Maximum Product of Three Numbers  max prod 3 num
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

# LC1299. Replace Elements with Greatest Element on Right Side - next greatest on right
def replaceElements(self, arr: List[int]) -> List[int]:
    mx = -1
    for i in range(len(arr))[::-1]:
        arr[i], mx = mx, max(mx, arr[i])
    return arr

# LC414. Third Maximum Number 3rd max
def thirdMax(self, nums: List[int]) -> int:
    max1 = max2 = max3 = -inf
    for n in nums:
        if n > max1:
            max1, max2, max3 = n, max1, max2
        elif max1 > n > max2:  # exclude max1 = n dupe
            max2, max3 = n, max2
        elif max2 > n > max3: max3 = n
    return max1 if max3 == -inf else max3

# LC136. Single Number - single nodupe in array, others are 2 single one
def singleNumber(self, nums: List[int]) -> int:
    res = 0
    for i in nums: res ^= i
    return res

# LC137. Single Number II  3 times except one three times
def singleNumber(self, nums: List[int]) -> int:
    seen_once = seen_twice = 0  # add all bits on ith, module 3, that's the loner
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
                                                      # [1, 0], 0th bit 1 seen odd times
    return seen_once  # [2, 2, 3, 2] -> [2, 0], [0, 2], [1, 0], [3, 0] -> 3

# LC217. Contains Duplicate  appear dupe contains dupe contain dupe at least twice
def containsDuplicate(self, nums: List[int]) -> bool:
    showed = set()
    for i in nums:
        if i in showed: return True
        showed.add(i)
    return False
def containsDuplicate(self, nums):
    return len(set(nums)) < len(nums)

# LC645. Set Mismatch set int duped repetition of one number
def findErrorNums(self, nums: List[int]) -> List[int]:
    dupe = missing = -1
    for n in nums:
        if nums[abs(n)-1] < 0: dupe = abs(n)
        else: nums[abs(n)-1] *= -1
    missing = next((i+1 for i, n in enumerate(nums) if n > 0), -1)
    return [dupe, missing]

# LC485. Max Consecutive Ones   max count 1 max 1 count
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



# LC775. Global and Local Inversions
def isIdealPermutation(self, A):
    return all(abs(i-x) <= 1 for i,x in enumerate(A))

# LC679. 24 Game card 24 card
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

# LC315. Count of Smaller Numbers After Self  number of smaller to right small to right
def countSmaller(self, nums: List[int]) -> List[int]:
    from sortedcontainers import SortedList  # AVL tree implementation or red black tree
    n = len(nums)  # O(nlogn) time, O(n) space
    ans = [0] * n
    x = SortedList()
    for i, val in enumerate(reversed(nums)):  # O(n)
        ans[~i] = x.bisect_left(val)  # O(logn)  ~i = n-1-i
        x.add(val)  # O(logn), in sorted order
    return ans
class BIT:  # Binary Indexed Tree (Fenwick Tree)
    def __init__(self, n):
        self.ans = [0] * (n+1)
    def query(self, i):
        total = 0
        while i:
            total += self.ans[i]
            i -= i & -i
        return total
    def update(self, i, val):
        while i < len(self.ans):
            self.ans[i] += val # prefix sum
            i += i & -i  # add rightmost bit to go up
class Solution:
    def countSmaller(self, nums):
        n, answer = len(nums), []
        dict1 = {j: i+1 for i, j in enumerate(sorted(nums))}
        result = BIT(n)
        for i in range(n-1,-1,-1):
            answer.append(result.query(dict1[nums[i]]-1))
            result.update(dict1[nums[i]], 1)
        return answer[::-1]
# https://leetcode.com/problems/range-sum-query-mutable/solutions/75753/Java-using-Binary-Indexed-Tree-with-clear-explanation/
def countSmaller(self, nums: List[int]) -> List[int]:
    sorted_arr = [] # O(nlogn) time and O(n) space (test shows this is much slower)
    rst = []
    for num in nums[::-1]:
        idx = bisect_left(sorted_arr, num)
        rst.append(idx)
        sorted_arr.insert(idx, num)  # this is o(n)
    return rst[::-1]

# LC493. Reverse Pairs
def reversePairs(self, nums: List[int]) -> int:  # O(nlogn)
    import sortedcontainers
    arr = sortedcontainers.SortedList()
    count = 0
    for i in reversed(range(len(nums))):
        count += arr.bisect_left(nums[i])
        arr.add(2*nums[i])
    return count

# LC1470. Shuffle the Array - merge alternatively  shuffle array
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



# LC350. Intersection of Two Arrays II - same elems appear multiple times, intersection 2 array
def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:  # O(n + m)
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

# LC1802. Maximum Value at a Given Index in a Bounded Array constructed array
def maxValue(self, n: int, index: int, maxSum: int) -> int:
    maxSum -= n  # this is the trick!!! get rid of 1s paddings
    l, r = 0, maxSum
    while l <= r:
        m = (l + r) // 2
        s = max(0, m - index - 1)
        e = max(0, m - n + index)
        # height of m triangles - left/right corner triangles
        # these 2 corners are formed by boundaries
        if m * m - s * (s + 1) // 2 - e * (e + 1) // 2 <= maxSum: l = m + 1
        else: r = m - 1
    return r + 1
# https://leetcode.com/problems/maximum-value-at-a-given-index-in-a-bounded-array/solutions/3643443/python-binary-search-well-explained-deep-explanation-i-passed-hours-to-fully-understand/?envType=company&envId=apple&favoriteSlug=apple-more-than-six-months
# https://leetcode.com/problems/maximum-value-at-a-given-index-in-a-bounded-array/?envType=company&envId=apple&favoriteSlug=apple-more-than-six-months

# LC1732. Find the Highest Altitude
def largestAltitude(self, gain: List[int]) -> int:
    h = 0
    mh = 0 # this is wrong, should -inf
    for g in gain:
        h += g
        mh = max(mh, h)
    return mh

# LC1232. Check If It Is a Straight Line
def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
    (x1, y1), (x2, y2) = coordinates[0], coordinates[1]
    dy, dx = y2 - y1, x2 - x1
    for x3, y3 in coordinates[2:]:
        if dy * (x3 - x1) != (y3 - y1) * dx: return False
    return True

# LC135. Candy alloc candy assign
def candy(self, ratings: List[int]) -> int:
    n = len(ratings)
    candies = [1] * n
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)
    return sum(candies)
# https://leetcode.com/problems/candy/solutions/4037646/99-20-greedy-two-one-pass/
def candy(self, ratings: List[int]) -> int:
    ret, up, down, peak = 1, 0, 0, 0
    for prev, curr in zip(ratings[:-1], ratings[1:]):
        if prev < curr:
            up, down, peak = up + 1, 0, up + 1
            ret += 1 + up
        elif prev == curr:
            up = down = peak = 0
            ret += 1
        else:
            up, down = 0, down + 1
            ret += 1 + down - int(peak >= down)
    return ret
# https://leetcode.com/problems/candy/solutions/4037646/99-20-greedy-two-one-pass/

# LC2491. Divide Players Into Teams of Equal Skill  chemistry prod
def dividePlayers(self, skill: List[int]) -> int:
    n = len(skill)
    total_skill = sum(skill)
    if total_skill % (n // 2) != 0: return -1
    target_skill = total_skill // (n // 2)
    skill_map = Counter(skill)
    total_chemistry = 0
    # Iterate through unique skill values
    for curr_skill, curr_freq in skill_map.items():
        partner_skill = target_skill - curr_skill
        if partner_skill not in skill_map or curr_freq != skill_map[partner_skill]:
            return -1
        total_chemistry += curr_skill * partner_skill * curr_freq
    # Return half of total chemistry (as each pair is counted twice)
    return total_chemistry // 2

# LC3326. Minimum Division Operations to Make Array Non Decreasing  min div ops
def minOperations(self, nums: List[int]) -> int:  # O(n*sqrt(max(nums)) time
    @cache
    def least_prime(num: int) -> int:
        if num % 2 == 0: return 2
        for prime in range(3, math.isqrt(num)+1, 2):
            if num % prime == 0: return prime
        return num
    ans = 0
    for i in range(len(nums) - 2, -1, -1):
        if nums[i] <= nums[i + 1]: continue
        nums[i] = least_prime(nums[i])
        if nums[i] > nums[i + 1]: return -1
        ans += 1
    return ans
