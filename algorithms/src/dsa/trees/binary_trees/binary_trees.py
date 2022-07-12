# https://www.careercup.com/question?id=5765256088387584

# Given a binary tree that complies to the following rule: The parent node value is always the
# result of the AND operator of its children values.
# You may modify one of the leaf node values. Write a function to fix the tree to comply the
# above rule

def fix_tree(node):
    if not node:
        return 1
    if not node.left and not node.right:
        return node.value
    node.value = fix_tree(node.left) & fix_tree(node.right)
    return node.value


