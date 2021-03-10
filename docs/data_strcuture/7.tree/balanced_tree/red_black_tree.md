# Red-Black Tree

Java HashSet is implemented with the Red-Black tree.

Red-Black trees, unlike AVL tree, requires at most 2 rotations to balance the tree
after insertion and deletion.

While AVL tree is strictly balanced, RB tree is only strictly balanced on black nodes.
AVL search is thus faster than RB, but slow on insertion and deletion.

RB tree has the following properties:
- Every node is black or red, root is black
- Every NIL leaf is treated as black
- If a node is red, then its children are black
- Every tree path from a node to NIL leaves have same number of black nodes.

## Insertion:

https://www.youtube.com/watch?v=qA02XWRTBdw


- create a new RED node with the value and insert it into the BST
- if parent is RED, change the color to BLACK
