

stack/queue interface:
push, pop, peek

LC 232 How to implement a queue with stacks?
Use 2 stacks to pop out
```
def enqueue(e):
    stack1.push(e)
    
# amortized is O(1), i.e., n operations take O(n) time.
def dequeue():  
    if not stack2:
        while stack1:
            stack2.push(stack1.pop())
    
    return stack2.pop()
```

LC 225 How to implement a stack with queues?
This is not as elegant as above. To reverse a queue, use recursion
```
def qcopy(qin, qout):
    if qin:
        e = qin.popleft()
        qcopy(qin, qout)
        qout.append(e)
```


https://introcs.cs.princeton.edu/java/43stack/
