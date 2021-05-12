# We have N different apps with different user growth rates. At a given time t, measured in days, the number of users
# using an app is g^t (for simplicity we'll allow fractional users), where g is the growth rate for that app. These
# apps will all be launched at the same time and no user ever uses more than one of the apps. We want to know how many
# total users there are when you add together the number of users from each app.
# After how many full days will we have 1 billion total users across the N apps?
# Signature
# int getBillionUsersDay(float[] growthRates)
# Input
# 1.0 < growthRate < 2.0 for all growth rates
# 1 <= N <= 1,000
# Output
# Return the number of full days it will take before we have a total of 1 billion users across all N apps.
# Example 1
# growthRates = [1.5]
# output = 52
# Example 2
# growthRates = [1.1, 1.2, 1.3]
# output = 79
# Example 3
# growthRates = [1.01, 1.02]
# output = 1047
import math

# Add any helper functions you may need here


def getBillionUsersDay(growthRates):
  # Write your code here
  start, end = 1, 2000
  th = 10 ** 9
  while start < end:  # bisect on days
    mid = (start + end) / 2
    tot = 0
    for r in growthRates:
      tot += r ** mid

    if tot > th:
      end = mid - 1
    else:
      start = mid
  return math.ceil(start)
