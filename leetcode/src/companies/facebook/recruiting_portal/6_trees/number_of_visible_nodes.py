# There is a binary tree with N nodes. You are viewing the tree from its left side and can see only the leftmost nodes at each level. Return the number of visible nodes.
# Note: You can see only the leftmost nodes, but that doesn't mean they have to be left nodes. The leftmost node at a level could be a right node.
# Signature
# int visibleNodes(Node root) {
# Input
# The root node of a tree, where the number of nodes is between 1 and 1000, and the value of each node is between 0 and 1,000,000,000
# Output
# An int representing the number of visible nodes.
# Example
#
#             8  <------ root
#            / \
#          3    10
#         / \     \
#        1   6     14
#           / \    /
#          4   7  13
#
# output = 4
def visible_nodes(root):
  # Write your code here
  ret = []
  queue = [root]
  while queue:
    #for a in queue: print(a.val)
    # print('-'*20)
    ret.append(queue[0])
    queue = [kid for n in queue for kid in (n.left, n.right) if kid]
    #for a in ret: print(a.val)
    #print('='*20)
  return len(ret)
