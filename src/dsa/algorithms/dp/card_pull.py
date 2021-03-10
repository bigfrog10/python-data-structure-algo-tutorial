# Amazon SDE-3s
# As we all know that poker cards have 4 suites: Spades, Hearts, Clubs and Diamonds
# with figures from 1 to 13. Now you are given a set of poker cards(not necessarily
# the full set), you can pick any one card as the first card. And except for the
# first card, you can only pick cards that have same suites or sam figure with the
# previous one. Return the max number of cards you can.

# can be viewed as the largest strongly connected component.

# DFS for max, O(n^2)

