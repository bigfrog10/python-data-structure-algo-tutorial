# The Python itertools.chain() method generates an iterator from multiple iterables.
# This simply chains all the iterables together into one sequence and returns a single
# iterator to that combined sequence. The syntax for this method is as follows
#    iterator = itertools.chain(*sequence)
import itertools

list1 = ['hello', 'from', 'AskPython']
list2 = [10, 20, 30, 40, 50]
dict1 = {'site': 'AskPython', 'url': 'https://askpython.com'}

for item in itertools.chain(list1, list2, dict1):
    print(item)

# itertools.cycle()
# itertools.tee()

# itertools.takewhile(lambda, list)
# itertools.dropwhile(lambda, list)
# accumulate(list, lambda)
# starmap(lambda, list)

# permutations()
# combinations()
# combinations_with_replacement()

# product()
# groupby()

# itertools.count() is similar to range
# itertools.islice()
# itertools.repeat()


