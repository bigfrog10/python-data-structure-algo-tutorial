Suppose we have an algorithm has runtime O(N^α). Now we want to estimate α.
We may assume the runtime is xN^α + y.

If input with size N take some time s, then we vary the size N by 10 times and 
100 times:

input size N: xN^α + y = s  
input size 10N: x(10N)^α + y = t  
input size 100N: x(100N)^α + y = u  

Here, s, t, u are observed numbers when we run the program. Now subtract 
first 2 equations from last,

```(10^α - 1)xN^α = t - s```

```(100^α - 1)xN^α = u - s```

Divide them,

```(100^α - 1) / (10^α - 1) = (u - s) / (t - s)```

```10^α + 1 = (u - s) / (t - s)```

```10^α = (u - s) / (t - s) - 1 = (u - t) / (t - s)```

```α = log[(u - t) / (t - s)]```

For example,
input is 2134, 21343, 213431, roughly 10 times larger in next steps.
time taken are 30, 130, 3092. Then

```α = log[(u - t) / (t - s)] = log[(3092 - 130) / (130 - 30)] = 1.45```

So this is roughly O(N^1.5).

