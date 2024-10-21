from typing import List
import heapq

# LC646. Maximum Length of Pair Chain
def findLongestChain(self, pairs: List[List[int]]) -> int:
    # greedy on finish time so we could squeeze more intervals to get max number of intervals
    pairs.sort(key=lambda x: x[1]) #
    count, curr = 0, float('-inf')
    for left, right in pairs:
        if curr < left:
            curr = right
            count += 1
    return count

# LC798. Smallest Rotation with Highest Score - digit rotation small
def bestRotation(self, nums: List[int]) -> int:  # O(n) time and space
    N = len(nums)
    bad = [0] * N
    for i, x in enumerate(nums):
        left, right = (i - x + 1) % N, (i + 1) % N  # # how many shifts are bad, not indices
        bad[left] -= 1
        bad[right] += 1
        if left > right:
            bad[0] -= 1
    best = -N
    ans = cur = 0
    for i, score in enumerate(bad):
        cur += score
        if cur > best:
            best = cur
            ans = i
    return ans

# LC56. Merge Intervals - remove overlaps of a list
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])  # O(nlogn)
    merged = []
    for intv in intervals:
        if not merged or merged[-1][1] < intv[0]: merged.append(intv)  # no overlap
        else: merged[-1][1] = max(merged[-1][1], intv[1])
    return merged

# LC1024. Video Stitching - greedy
def videoStitching(self, clips: List[List[int]], T: int) -> int:  # O(Tn) time and O(T) space
    maxv = max(c[1] for c in clips)
    max_jumps = [0]*(maxv + 1)
    for l,r in clips: max_jumps[l] = max(max_jumps[l], r)
    res = lo = hi = 0  # it is then a jump game
    while hi < T:
        lo, hi = hi, max(max_jumps[lo:hi+1])
        if hi <= lo: return -1
        res += 1
    return res
def videoStitching(self, clips, T):  # O(nlogn) time and O(1) space
    end, end2, res = -1, 0, 0
    for i, j in sorted(clips):
        if end2 >= T or i > end2: break
        elif end < i <= end2:
            res, end = res + 1, end2
        end2 = max(end2, j)
    return res if end2 >= T else -1

# LC759. Employee Free Time
class Interval:
    def __init__(self, start: int = None, end: int = None):
        self.start = start
        self.end = end
def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':  # O(nlogm), m = # of employees
    # merge to a single list and merge-sorted by start, O(n), not O(nlogn)
    all_s = heapq.merge(*schedule, key=lambda x: x.start)
    # [[[1,2],[5,6]],[[1,3]],[[4,10]]] to [[1, 2], [1, 3], [4, 10], [5, 6]]
    ans = []
    prev = next(all_s).end
    while a := next(all_s, None):
        if a.start > prev:
            ans.append(Interval(prev, a.start))
        prev = max(prev, a.end)
    return ans

# LC636. Exclusive Time of Functions, jobs, process time, cpu single thread
def exclusiveTime(self, n, logs):  # O(n) runtime and space
    res, stack = [0] * n, []
    for log in logs:
        log = log.split(":")  # function id, start/end, timestamp
        if log[1] == "start": stack.append([int(log[2]), 0])  # 0 means no time spent on this yet.
        else:  # end
            start = stack.pop()
            time = int(log[2]) - start[0] + 1  # time spent on this pid
            res[int(log[0])] += time - start[1]  # add exclusive time, no child time.
            if stack: stack[-1][1] += time  # update parent time
    return res



# LC986. Interval List Intersections - of 2 lists of intervals  interval intersaction
def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    ret = []  # O(m + n)
    i = j = 0
    while i < len(firstList) and j < len(secondList):
        left = max(firstList[i][0], secondList[j][0])
        right = min(firstList[i][1], secondList[j][1])
        if left <= right: ret.append((left, right))  # add intersection
        if firstList[i][1] < secondList[j][1]: i += 1  # smaller start can enclose bigger start
        else: j += 1
    return ret

# LC252. Meeting Rooms - can attend all meetings?
def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
    si = sorted(intervals)  # sort by first element in asc order
    for i in range(len(si) - 1):
        if si[i][1] > si[i+1][0]: return False
    return True

# LC253. Meeting Rooms II - min # of conf rooms asked
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    intervals.sort()  # greedy, sort intervals by starting time. O(nlogn)
    rooms = []  # end times
    for intv in intervals:
        if rooms and rooms[0] <= intv[0]: # if earliest end time < this start time
            heapq.heappop(rooms)  # remove and replace with current end time
        heapq.heappush(rooms, intv[1])  # we sort heap by end time
    return len(rooms)

# LC1288. Remove Covered Intervals
def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
    intervals.sort(key = lambda x: (x[0], -x[1]))
    count = prev_end = 0
    for _, end in intervals:
        if end > prev_end:  # if current interval is not covered by the previous one
            count += 1
            prev_end = end
    return count

# LC370. Range Addition
def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
    arr = [0] * length  # O(n), use (start, end) as signals
    for update in updates:
        arr[update[0]] += update[2]
        if update[1] + 1 < length: arr[update[1] + 1] -= update[2]
    for i in range(1, length):
        arr[i] += arr[i-1]
    return arr

# LC465. Optimal Account Balancing
def minTransfers(self, transactions: List[List[int]]) -> int:  # O(2^N)
    tuplify = lambda balance: tuple(sorted((k, v) for k, v in balance.items()))
    @lru_cache(None)
    def dfs(balances):
        if not balances: return 0
        res = math.inf
        balances = {k: v for k, v in balances}
        for size in range(2, len(balances) + 1):
            for group in itertools.combinations(balances.keys(), size):
                if sum(balances[k] for k in group) == 0:
                    remaining_balances = {k: v for k, v in balances.items() if k not in group}
                    res = min(res, size - 1 + dfs(tuplify(remaining_balances)))
        return res

    balances = collections.defaultdict(int)
    for u, v, z in transactions:
        balances[u] += z
        balances[v] -= z
    return dfs(tuplify({k: v for k, v in balances.items() if v}))

# LC1523. Count Odd Numbers in an Interval Range
def countOdds(self, low: int, high: int) -> int:
    return (high + 1) // 2 - low // 2
    # the count of odd numbers between 1 and low - 1 is low // 2
    # the count of odd numbers between 1 and high is (high + 1 ) // 2








# LC630. Course Schedule III
def scheduleCourse(self, A: List[List[int]]) -> int:  # nlogn
    start, pq = 0, []  # pq has all courses taken now
    for t, end in sorted(A, key = lambda c: c[1]):  # sort by deadline
        start += t
        heapq.heappush(pq, -t)  # finish shortest duration first to save more room for others
        while start > end:  # if can't take, pop out
            start += heapq.heappop(pq)
    return len(pq)





# LC435. Non-overlapping Intervals - minimal intervals to be removed for nonoverlapping
def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    # https://leetcode.com/problems/non-overlapping-intervals/discuss/276056/Python-Greedy-Interval-Scheduling
    # O(nlogn) since we sort the input
    # greedy, make more room for later
    # other optimal solution can be replaced with this one.
    end, cnt = float('-inf'), 0
    for s, e in sorted(intervals, key=lambda x: x[1]):  # sort by end time
        if s >= end: end = e  # no overlap, keep it, so update new ending
        else:  # new interval starting < current end, so overlapped
            cnt += 1  # overlapped, so remove this, so increment counter
    return cnt

# LC729. My Calendar I
from sortedcontainers import SortedList
class MyCalendar:
    def __init__(self): self.arr = SortedList()  # list is not good enough for performance
    def book(self, start, end):  # O(logn) for 1 booking
        q1, q2 = self.arr.bisect_right(start), self.arr.bisect_left(end)  #logn
        if q1 == q2 and q1 % 2 == 0:
            self.arr.add(start)  # logn
            self.arr.add(end)    # logn
            return True
        return False

# LC731. My Calendar II
class MyCalendarTwo:
    def __init__(self):
        self.pos = [0, 10**9] # to deal with boundary case.
        self.cnt = {0: 0, 10**9: 0}
    def book(self, start: int, end: int) -> bool:  # O(n) due to for loops
        i = bisect.bisect_left(self.pos, start)
        j = bisect.bisect_left(self.pos, end)
        if any(self.cnt[self.pos[k]] >= 2 for k in range(i, j)): return False # triple booking
        if start not in self.cnt:
            c = self.cnt[self.pos[i-1]] #if i-1 >= 0 else 0
            if c >= 2: return False
            self.pos[i: i] = [start]  # insertion
            self.cnt[start] = c
            j += 1 # insert start at i, so shift j by 1
        if end not in self.cnt:
            self.pos[j: j] = [end]  # insertion
            self.cnt[end] = self.cnt[self.pos[j-1]]
        for k in range(i, j): self.cnt[self.pos[k]] += 1
        return True

# LC732. My Calendar III k-booking
import sortedcontainers
class MyCalendarThree1:
    def __init__(self):
        self.timeline = sortedcontainers.SortedDict([(-1, 0)])
        self.most = 0
    def book(self, start: int, end: int) -> int:
        # if keys are missing, set the value to previous values
        sidx = self.timeline.bisect_left(start)
        self.timeline[start] = self.timeline.get(start, self.timeline.peekitem(sidx-1)[1])
        eidx = self.timeline.bisect_left(end)  # we have to bisect after start
        self.timeline[end] = self.timeline.get(end, self.timeline.peekitem(eidx-1)[1])
        for i in range(sidx, eidx):  # no need to bisect again
            t, k = self.timeline.peekitem(i)
            self.timeline[t] = k+1
            self.most = max(self.most, k+1)
        return self.most
class MyCalendarThree(object):  # or binary index tree/segment tree
    def __init__(self):
        self.pos = []
        self.delta = {}  # count
        self.max = 0
    def book(self, start, end):  # O(logn) to O(n)
        i = bisect.bisect_left(self.pos, start)
        if start not in self.delta:
            self.delta[start] = self.delta[self.pos[i-1]] if i else 0
            self.pos[i:i] = [start]
        j = bisect.bisect_left(self.pos, end)
        if end not in self.delta:
            self.delta[end] = self.delta[self.pos[j-1]]
            self.pos[j:j] = [end]
        for k in range(i, j):
            self.delta[self.pos[k]] = c = self.delta[self.pos[k]] + 1
            self.max = max(self.max, c)
        return self.max

# LC715. Range Module
from bisect import bisect_left as bl, bisect_right as br
class RangeModule:
    def __init__(self):
        self._X = []  # intervals are [even, odd)
    def addRange(self, left, right):
        i, j = bl(self._X, left), br(self._X, right)
        self._X[i:j] = [left]*(i % 2 == 0) + [right]*(j % 2 == 0)  # O(n)
    def queryRange(self, left, right):
        i, j = br(self._X, left), bl(self._X, right)
        return i == j and i%2 == 1
    def removeRange(self, left, right):
        i, j = bl(self._X, left), br(self._X, right)
        self._X[i:j] = [left]*(i%2 == 1) + [right]*(j%2 == 1)  # O(n)
class RangeModule:
    def __init__(self):
        self.starts = SortedDict({-1: 0})  # intervals start -> end
    def addRange(self, left: int, right: int) -> None:  # O(klogn)
        while True:
            pos = self.starts.bisect_right(right)
            mkey = self.starts.iloc[pos-1]
            if mkey == -1: break
            mval = self.starts[mkey]
            if mval < left: break
            left, right = min(left, mkey), max(right, mval)
            del self.starts[mkey]
        self.starts[left] = right
    def queryRange(self, left: int, right: int) -> bool:  # logn
        pos = self.starts.bisect_right(left)
        mkey = self.starts.iloc[pos-1]
        return mkey > -1 and self.starts[mkey] >= right
    def removeRange(self, left: int, right: int) -> None:  # O(klogn)
        self.addRange(left, right)
        pos = self.starts.bisect_right(left)
        mkey = self.starts.iloc[pos-1]
        mval = self.starts[mkey]
        del self.starts[mkey]
        if mkey < left: self.starts[mkey] = left
        if mval > right: self.starts[right] = mval

# LC57. Insert Interval - nonoverlapping insert, interval insert
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    s, e = newInterval[0], newInterval[1]  # O(n)
    left = [i for i in intervals if i[1] < s]
    right = [i for i in intervals if i[0] > e]
    if left + right != intervals:  # we have overlaps, update start and end
        s = min(s, intervals[len(left)][0])
        e = max(e, intervals[~len(right)][1])
    return left + [[s, e]] + right



# LC1353. Maximum Number of Events That Can Be Attended max event attend
def maxEvents(self, events: List[List[int]]) -> int:  # O(nlogn)
    events.sort(reverse=1)  # should sort by start date, we do this for pop()
    hq = []  # store end time of open events
    res = d = 0  # d is current day
    while events or hq:
        if not hq: d = events[-1][0]  # current day is start day
        while events and events[-1][0] <= d:  # push all events we can possibly attend
            heapq.heappush(hq, events.pop()[1])  # sort by end date
        heapq.heappop(hq)  # attend this one event with earlist end date, save room for more future
        res += 1
        d += 1
        while hq and hq[0] < d: heapq.heappop(hq)  # remove all impossible-to-attend events
    return res

# LC1751. Maximum Number of Events That Can Be Attended II - with max meeting values
def maxValue(self, events: List[List[int]], k: int) -> int:
    n = len(events)
    events.sort(key=lambda x: x[1])

    # k is the number of events we can attend
    # end is the last event we attended's END TIME
    # event_index is the current event we are checking
    @lru_cache(None)
    def dp(end: int, event_index: int, k: int):
        if k == 0 or event_index == n: return 0
        event_start, event_end, event_value = events[event_index]
        # Can we attend this event?
        # Does its start time conflict with the previous events end time?
        # If the start time is the same as the end time we cannot end as well (view example 2)
        if event_start <= end: # Could not attend, check the next event
            return dp(end, event_index + 1, k)
        attend = event_value + dp(event_end, event_index + 1, k - 1)
        skip = dp(end, event_index + 1, k)
        return max(attend, skip)
    return dp(0, 0, k)

# LC1094. Car Pooling
def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
    # bucket sort, so we have O(n) since we have upper bound
    timestamps = [0] * (max(x[2] for x in trips) + 1)
    for c, s, e in trips:
        timestamps[s] += c
        timestamps[e] -= c
    cap = 0
    for ts in timestamps:
        cap += ts
        if cap > capacity: return False
    return True

# LC452. Minimum Number of Arrows to Burst Balloons
def findMinArrowShots(self, points: List[List[int]]) -> int:
    if not points: return 0
    points.sort(key=lambda x: x[1])  # sort by end time, O(nlogn)
    cnt, end = 1, points[0][1]
    for s, e in points:
        if s > end:  # finished before next start, so we need a new arrow
            cnt += 1
            end = e
    return cnt

# LC1854. Maximum Population Year
def maximumPopulation(self, logs: List[List[int]]) -> int:  # O(n)
    delta, start = [0] * 101, 1950  # the timespan 1950-2050 covers 101 years
    for l in logs:  # counting sort
        delta[l[0] - start] += 1
        delta[l[1] - start] -= 1
    cumu, maxPop, year = 0, 0, start
    for i, d in enumerate(delta):
        cumu += d
        if cumu > maxPop:
            maxPop, year = cumu, start + i
    return year

# LC1169. Invalid Transactions
class Solution:
    def invalidTransactions(self, transactions: List[str]) -> List[str]:
        counter = Counter(transactions)
        tx_cache = defaultdict(list)
        ret = set()
        for tx in transactions:
            name, ts, amount, city = tx.split(',')
            invalid = False
            if int(amount) > 1000: invalid = True
            hist = tx_cache[name]
            ts = int(ts)
            idx = bisect.bisect_left(hist, (ts, city, tx))
            i = idx-1
            while i >= 0 and ts - hist[i][0] <= 60:
                if hist[i][1] != city:
                    ret.add(hist[i][2])
                    invalid = True
                i -= 1
            i = idx
            while i < len(hist) and hist[i][0] - ts <= 60:
                if hist[i][1] != city:
                    ret.add(hist[i][2])
                    invalid = True
                i += 1
            if invalid: ret.add(tx)
            hist.insert(idx, (ts, city, tx))
            #bisect.insort(hist, (ts, city, tx))
        ret1 = []
        for tx in ret: ret1.extend([tx] * counter[tx])
        return ret1

# LC1029. Two City Scheduling
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        # Sort by a gain which company has by sending a person to city A and not to city B
        costs.sort(key = lambda x : x[0] - x[1])  # greedy
        total = 0
        n = len(costs) // 2
        # To optimize the company expenses, send the first n persons to the city A
        # and the others to the city B
        for i in range(n): total += costs[i][0] + costs[i + n][1]
        return total

# LC1235. Maximum Profit in Job Scheduling - backpack
def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
    # sort rows by start time, total O(nlogn)
    jobs = sorted([v for v in zip(startTime, endTime, profit)], key=lambda x: x[0])
    start, end, profit = zip(*jobs) # unpack after sorting
    @lru_cache(None)
    def dp(i):  # profits from i and after
        if i == len(start): return 0
        j = bisect_left(start, end[i])
        return max(profit[i] + dp(j), dp(i + 1))
    return dp(0)

# LC1229. Meeting Scheduler
def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
    # build up a heap containing time slots last longer than duration
    # If there is an overlap, the 2 slots must come from 2 people.
    timeslots = list(filter(lambda x: x[1] - x[0] >= duration, slots1 + slots2))
    heapq.heapify(timeslots)
    while len(timeslots) > 1:
        start1, end1 = heapq.heappop(timeslots)
        start2, end2 = timeslots[0]
        if end1 >= start2 + duration:
            return [start2, start2 + duration]
    return []

# https://www.codesdope.com/blog/article/find-maximum-meetings-in-one-room/
def fit_in_one_room(meetings: list) -> int:  # input is list of [begin, end]
    # greedy: allocate by end time to save more time for later
    meetings.sort(key=lambda x: x[1]) # sort meetings by end.
    res = 1 # first meeting
    end = meetings[0][1]
    for i in range(1, len(meetings)):
        if meetings[i][0] >= end:
            res += 1
            end = meetings[i][1]
    return res

print(fit_in_one_room([[1, 3], [2, 4], [3, 6]]))
