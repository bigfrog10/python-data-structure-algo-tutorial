
### 2's compliment:
```-x = ~x + 1```

```python
# LC476. Number Complement
def bitwiseComplement(self, N: int) -> int:
    if N == 0: return 1
    todo, bit = N, 1
    while todo:
        N = N ^ bit  # flip current bit
        bit = bit << 1  # prepare for the next run
        todo = todo >> 1
    return N
```
### XOR
```a ^ b = 0 if a == b else 1 ```

Properties:  
```
n ^ n = 0
n ^ 0 = n
n ^ 1 = n-1 if n is odd or n+1 if n is even (operates on last bit)
```

### Right Most Bit 
Given x, (x-1) will have all the bits same as x, except for the rightmost 1 in x 
and all the bits to the right of the rightmost 1.  
```110 & 101 = 100 for 6 & (6-1) = 4```

```RMB = x & -x```

To erase RMB
```x = x & (x-1)```

To check whether x is a power of 2:
```x & (x-1) == 0```

To count number of 1's in x: (O(k), worst is O(logn))
```
def ones(n):
    cnt = 0
    while n:
        n = n & (n-1) # this erases the rightmost bit to 0.
        cnt += 1
    return cnt
```
Both ```x & (-x)``` and ```x ^ ( x & (x-1) )``` gives the right most bit

### Most Significant Bit
For given x, to get MSB we fill 1's to x by n | (n >> 2 ** i) for i in 0 ... 6
```
def msb(n):
    if n == 0: return 0
    for i in range(6): # for 64 bits
        n = n | (n >> 2 ** i)
    return (n+1) >> 1  # since n now has all 1's starting with msb.
```

x = x ^ 1 ^ 1  # flip

### References:
- https://hackernoon.com/xor-the-magical-bit-wise-operator-24d3012ed821
- https://www.hackerearth.com/practice/notes/bit-manipulation/

