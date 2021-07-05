# https://leetcode.com/discuss/interview-question/346748/Bloomberg-or-Phone-Screen-or-File-Diffs-With-Ram-Constraints
# There are 2 files, A and B, each with size 100MB. Each file is composed of trades
# that take up one line each, and each trade has a unique key.
# You have 1MB memory. Now we need to print out all trades that are in only 1 file
# but not the other.

# N^2 is too slow.
# sort each file by id, use external merge sort
#     - sort each chunk and write to disk tmp files.
#     - merge k-sorted list, and write to disk
# compare sorted files, say, C and D
