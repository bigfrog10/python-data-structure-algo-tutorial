################################################################################
# Given an array where elements are sorted in ascending order, convert it to a
# height balanced BST.
# For this problem, a height-balanced binary tree is defined as a binary tree in
# which the depth of the two subtrees of every node never differ by more than 1.
# EASY
################################################################################
# BST: less than right node and greater than left node

from dataclasses import  dataclass
import math

@dataclass()
class TreeNode:
    val: int
    left: 'TreeNode'
    right: 'TreeNode'


def to_balanced_bst(nums: list) -> TreeNode:
    # min level such that len < 2 ** level - 1
    level = math.log2(len(nums) + 1)
    print(level)
    level = math.ceil(level)
    print(level)

    def to_bst(nums, b, e) -> TreeNode:
        if b > e:
            return None

        middle = b + (e - b + 1) // 2  # if len even, use 2nd
        root = TreeNode(nums[middle], None, None)
        root.left = to_bst(nums, b, middle-1)
        root.right = to_bst(nums, middle + 1, e)
        return root

    return to_bst(nums, 0, len(nums)-1)


# to_balanced_bst([-10,-3,0,5,9])
# to_balanced_bst([x for x in range(7)])
# to_balanced_bst([x for x in range(8)])
# to_balanced_bst([x for x in range(15)])
# to_balanced_bst([x for x in range(16)])

print(to_balanced_bst([-10,-3,0,5,9]))
print(to_balanced_bst([x for x in range(7)]))
# print(to_balanced_bst([x for x in range(8)]))
# print(to_balanced_bst([x for x in range(16)]))
# print(to_balanced_bst([x for x in range(17)]))
