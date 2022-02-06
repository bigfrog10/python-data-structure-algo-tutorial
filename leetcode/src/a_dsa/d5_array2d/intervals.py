from typing import List
import heapq
from heapq import *

# LC636. Exclusive Time of Functions, jobs, process
def exclusiveTime(self, n, logs):  # O(n) runtime and space
    res, stack = [0] * n, []
    for log in logs:
        log = log.split(":")
        if log[1] == "start": stack.append([int(log[2]), 0])  # 0 means no time spent on this yet.
        else:  # end
            start = stack.pop()
            time = int(log[2]) - start[0] + 1  # time spent on this pid
            res[int(log[0])] += time - start[1]  # add exclusive time, no child time.
            if stack: stack[-1][1] += time  # update parent time
    return res

# LC56. Merge Intervals, top100
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]: merged.append(interval)  # no overlap
        else: merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

# LC759. Employee Free Time
class Interval:
    def __init__(self, start: int = None, end: int = None):
        self.start = start
        self.end = end
def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':  # fastest
    intervals = sorted((interval.start, interval.end) for s in schedule for interval in s)
    res, moving_end = [], intervals[0][1]
    for s, e in intervals:  # O(n)
        if s > moving_end: res.append(Interval(moving_end, s))
        moving_end = max(e, moving_end)
    return res

# LC435. Non-overlapping Intervals
def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    # https://leetcode.com/problems/non-overlapping-intervals/discuss/276056/Python-Greedy-Interval-Scheduling
    # O(nlogn) since we sort the input
    # greedy, make more room for later
    # other optimal solution can be replaced with this one.
    end, cnt = float('-inf'), 0
    for s, e in sorted(intervals, key=lambda x: x[1]):
        if s >= end: end = e  # no overlap, keep it, so update new ending
        else:  # new interval starting < current end, so overlapped
            cnt += 1  # overlapped, so remove this, so increment counter
    return cnt

# LC729. My Calendar I
class MyCalendar:
    def __init__(self): self.intervals = []  # like [s1, e1, s2, e2, ...]
    def book(self, start: int, end: int) -> bool:
        if end <= start: return False
        i = bisect.bisect_right(self.intervals, start)
        if i % 2: return False  # start is in some stored interval
        j = bisect.bisect_left(self.intervals, end)
        if i != j: return False  # overlapped
        self.intervals[i:i] = [start, end]
        return True

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

# LC732. My Calendar III
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

# LC986. Interval List Intersections
def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    ret = []
    i = j = 0
    while i < len(firstList) and j < len(secondList):
        left = max(firstList[i][0], secondList[j][0])
        right = min(firstList[i][1], secondList[j][1])
        if left <= right: ret.append((left, right))  # add intersection
        if firstList[i][1] < secondList[j][1]: i += 1  # move short end
        else: j += 1
    return ret

# LC715. Range Module
from bisect import bisect_left as bl, bisect_right as br
class RangeModule:
    def __init__(self):
        self._X = []  # intervals are [even, odd)
    def addRange(self, left, right):
        i, j = bl(self._X, left), br(self._X, right)
        self._X[i:j] = [left]*(i % 2 == 0) + [right]*(j % 2 == 0)
    def queryRange(self, left, right):
        i, j = br(self._X, left), bl(self._X, right)
        return i == j and i%2 == 1
    def removeRange(self, left, right):
        i, j = bl(self._X, left), br(self._X, right)
        self._X[i:j] = [left]*(i%2 == 1) + [right]*(j%2 == 1)

# LC252. Meeting Rooms
def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
    si = sorted(intervals)  # sort by first element in asc order
    for i in range(len(si) - 1):
        if si[i][1] > si[i+1][0]: return False
    return True

# LC253. Meeting Rooms II, top100
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    if not intervals: return 0
    intervals.sort()  # greedy, sort intervals by starting time. O(nlogn)
    rooms = []
    for intv in intervals:
        if rooms and rooms[0] <= intv[0]: # if earliest end time < this start time
            heapq.heappop(rooms) # remove and replace with current end time
        heapq.heappush(rooms, intv[1])  # we sort heap by end time
    return len(rooms)

# LC57. Insert Interval
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    s, e = newInterval[0], newInterval[1]
    left = [i for i in intervals if i[1] < s]
    right = [i for i in intervals if i[0] > e]
    if left + right != intervals:
        s = min(s, intervals[len(left)][0])
        e = max(e, intervals[~len(right)][1])
    return left + [(s, e)] + right

# LC370. Range Addition
def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
    arr = [0] * length  # O(n), use (start, end) as signals
    for update in updates:
        arr[update[0]] += update[2]
        if update[1] + 1 < length: arr[update[1] + 1] -= update[2]
    for i in range(1, length):
        arr[i] += arr[i-1]
    return arr

# LC1353. Maximum Number of Events That Can Be Attended
def maxEvents(self, events: List[List[int]]) -> int:
    events.sort()  # sort by begin date
    pq = []  # store end time of open events
    count = d = 0 # d is current day
    i, n = 0, len(events)
    while i < n or pq:
        if not pq: d = events[i][0] # current day is start day
        while i < n and d >= events[i][0]: # push all events we can possibly attend
            heappush(pq, events[i][1])  # sort by end date
            i += 1  # finish earlier to attend more events
        heappop(pq)  # attend this one event
        count += 1
        d += 1
        while pq and pq[0] < d: heappop(pq)  # remove all impossible-to-attend events
    return count

# LC1751. Maximum Number of Events That Can Be Attended II
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

# LC465. Optimal Account Balancing
def minTransfers(self, transactions: List[List[int]]) -> int:
# https://leetcode.com/problems/optimal-account-balancing/discuss/95355/Concise-9ms-DFS-solution-(detailed-explanation)
    debt = collections.defaultdict(int)   # n! because T(n) = n*T(n-1)
    for t in transactions: # each persion's net
        debt[t[0]] -= t[2]
        debt[t[1]] += t[2]
    def dfs(s): # dfs on ids
        while s < len(debt) and debt[s]==0: s += 1
        if s == len(debt): return 0
        r = float('inf')
        for i in range(s+1, len(debt)):
            if debt[i]*debt[s] < 0:
                debt[i] += debt[s] # settle s with i
                r = min(r, 1 + dfs(s+1))
                debt[i] -= debt[s] # backtrack
        return r
    return dfs(0)

# LC1854. Maximum Population Year
def maximumPopulation(self, logs: List[List[int]]) -> int:
    dates = []
    for birth, death in logs:
        dates.append((birth, 1))
        dates.append((death, -1))
    dates.sort()
    population = max_population = max_year = 0
    for year, change in dates:
        population += change
        if population > max_population:
            max_population = population
            max_year = year
    return max_year

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
        costs.sort(key = lambda x : x[0] - x[1]) # greedy
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
    def dp(i):
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