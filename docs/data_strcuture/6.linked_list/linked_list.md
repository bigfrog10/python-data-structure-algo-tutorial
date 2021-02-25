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
