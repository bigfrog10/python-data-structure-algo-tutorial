# We keep track of the revenue Facebook makes every day, and we want to know on what days Facebook hits certain revenue milestones. Given an array of the revenue on each day, and an array of milestones Facebook wants to reach, return an array containing the days on which Facebook reached every milestone.
# Signature
# int[] getMilestoneDays(int[] revenues, int[] milestones)
# Input
# revenues is a length-N array representing how much revenue FB made on each day (from day 1 to day N). milestones is a length-K array of total revenue milestones.
# Output
# Return a length-K array where K_i is the day on which FB first had milestones[i] total revenue. If the milestone is never met, return -1.
# Example
# revenues = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# milestones = [100, 200, 500]
# output = [4, 6, 10]
# Explanation
# On days 4, 5, and 6, FB has total revenue of $100, $150, and $210 respectively. Day 6 is the first time that FB has >= $200 of total revenue.
from itertools import accumulate
import bisect

# Add any helper functions you may need here


def getMilestoneDays(revenues, milestones):
  # Write your code here
  cumu = list(accumulate(revenues))
  # print(cumu)
  ret = []
  for m in milestones:
    idx = bisect.bisect_left(cumu, m)
    if idx >= len(cumu):
      ret.append(-1)
    else:
      ret.append(idx+1)
  return ret
