# META: https://leetcode.com/discuss/interview-question/124616/Merge-two-interval-lists
# Given A and B two interval lists, A has no overlap inside A and B has no overlap
# inside B. Write the function to merge two interval lists, output the result with
# no overlap. Ask for a very efficient solution
# A naive method can combine the two list, and sort and apply merge interval in the
# leetcode, but is not efficient enough.
# For example,
# A: [1,5], [10,14], [16,18]
# B: [2,6], [8,10], [11,20]
#
# output [1,6], [8, 20]
# Time complexity: O(m+n)
# Space Complexity: O(1)
def merge_intervals(list1, list2):
    # assume they are sorted by start time
    res = []
    i = j = 0
    while i < len(list1) or j < len(list2):
        if i == len(list1):
            temp = list2[j]
            j += 1
        elif j == len(list2):
            temp = list1[i]
            i += 1
        elif list1[i][0] < list2[j][0]:
            temp = list1[i]
            i += 1
        else:
            temp = list2[j]
            j += 1
        if not res or res[-1][1] < temp[0]:
            res.append(temp)
        else:
            res[-1][1] = max(res[-1][1], temp[1])
    return res


print (merge_intervals([[1,5], [10,14], [16,18]], [[2,6], [8,10], [11,20]]))
print (merge_intervals([[1,15], [10,14], [16,18]], [[2,16], [8,10], [11,20]]))
print (merge_intervals([[1,15], [20,24], [36,48]], [[15,15], [28,30]]))
print (merge_intervals([[1,15], [20,24], [36,48]], [[15,15], [28,36]]))