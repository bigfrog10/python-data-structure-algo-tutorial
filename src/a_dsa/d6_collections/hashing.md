

[LC 706](https://leetcode.com/problems/design-hashmap/)
Design a hashmap

[LC 146](https://leetcode.com/problems/lru-cache/)
Design a LRU cache

[LC 460](https://leetcode.com/problems/lfu-cache/)
Design a LFU cache


## Hashing
separate chain implementation
https://www.cpp.edu/~ftang/courses/CS240/lectures/hashing.htm

Robinhood Hashing
https://www.sebastiansylvan.com/post/robin-hood-hashing-should-be-your-default-hash-table-implementation/
https://github.com/narendly/robinhoodhashing/blob/master/PROBLEM.md

integer hashing:
https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key

use a fixed size a for buckets, if load factor > 0.7, say, we pick a bigger number for buckets.
for each bucket, we have a max size, b, linked list. 
count all entries as load ratio, if this is > 0.7, rebalance it.
we save both (k, v) in the entry for rebalance

integer hashing algorithm is multiplicative hashing, see here
https://en.wikipedia.org/wiki/Hash_function
https://en.wikipedia.org/wiki/Universal_hashing
There are other algorithms for string hashing and vector hashing
hash needs to be evenly distributed

a close implementation is:
https://leetcode.com/problems/design-hashmap/discuss/1033321/JavaScript-Multiplicative-hashing-%2B-buckets-%2B-dynamic-size

Further reading shows that we should use robinhood hashing
https://en.wikipedia.org/wiki/Hash_table#Choosing_a_good_hash_function
because multiplicative hashing has clustering, not uniform
https://www.sebastiansylvan.com/post/robin-hood-hashing-should-be-your-default-hash-table-implementation/

here is an implementation in javascript
https://leetcode.com/problems/design-hashmap/discuss/803140/Python-Single-Array-%2B-Robinhood-Hashing
