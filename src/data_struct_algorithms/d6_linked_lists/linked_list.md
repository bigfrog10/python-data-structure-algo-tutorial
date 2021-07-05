https://gist.github.com/zed/5651186

- Linked Lists  
https://blog.csdn.net/weixin_37780776/article/details/108066174


- Double Linked Lists
- Cyclic Linked Lists	
Joseph' Problem
The position returned by josephus(n - 1, k) is adjusted because the  
recursive call josephus(n - 1, k) considers the original position 
k%n + 1 as position 1.  
```
def josephus(n: int, k: int) -> int:
    if n == 1:
        return 1
    else:
        return (josephus(n-1, k) + k - 1) % n + 1
```  

[LC 138](https://leetcode.com/problems/copy-list-with-random-pointer/)
Copy linked list with random pointers

LC 21 Merge Two Sorted Lists

LC 61 rotate list

### Traversal
LTCD160. Intersection of Two Linked Lists
LTCD2. Add Two Numbers
LTCD817. Linked List Components
LTCD369. Plus One Linked List

### Merge
LTCD21. Merge Two Sorted Lists
LTCD23. Merge k Sorted Lists

### Reverse
LTCD206. Reverse Linked List
LTCD92. Reverse Linked List II
LTCD234. Palindrome Linked List
LTCD25. Reverse Nodes in k-Group

### 
LTCD328. Odd Even Linked List
LTCD61. Rotate List
LTCD138. Copy List with Random Pointer

### Delete
LTCD237. Delete Node in a Linked List
LTCD203. Remove Linked List Elements
LTCD19. Remove Nth Node From End of List
LTCD83. Remove Duplicates from Sorted List
LTCD82. Remove Duplicates from Sorted List II - keep distinct

### Cyclic Lists
LTCD141. Linked List Cycle
LTCD142. Linked List Cycle II
LTCD708. Insert into a Sorted Circular Linked List

### Doubled Linked Lists
