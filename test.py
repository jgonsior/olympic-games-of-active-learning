import random
import timeit
from typing import Any, List


# efficient
def _list_difference() -> List[Any]:
    long_list = random.sample(range(0, 10000), 1000)
    short_list = random.sample(long_list, 5)

    short_set = set(short_list)
    result = [i for i in long_list if not i in short_set]


def _using_sets_only():
    long_list = random.sample(range(0, 10000), 1000)

    short_list = random.sample(long_list, 5)
    return list(set(long_list) - set(short_list))


def _using_other_way_round():
    l1 = random.sample(range(0, 10000), 1000)
    l2 = random.sample(l1, 5)
    return [l1.remove(m) for m in l1 if m in l2]


def stupid_way():
    l1 = random.sample(range(0, 10000), 1000)
    l2 = random.sample(l1, 5)
    return [i for i in l1 if not i in l2]


repetition = 10000

print(timeit.timeit(lambda: _list_difference(), number=repetition))
print(timeit.timeit(lambda: _using_sets_only(), number=repetition))
print(timeit.timeit(lambda: _using_other_way_round(), number=repetition))
print(timeit.timeit(lambda: stupid_way(), number=repetition))
