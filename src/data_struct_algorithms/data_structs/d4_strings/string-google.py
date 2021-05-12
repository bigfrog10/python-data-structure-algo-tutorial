# https://leetcode.com/playground/8zLNbxHQ

# https://leetcode.com/discuss/interview-experience/1177461/Google-Phone-Interview

# // Question:
#
# // Example input:
# // "some text"
# // [0, 4) / “some” -> X
# // [5, 8) / “tex” -> Y
# // [3, 6) / “e t” -> Z
# //
# // You need to output consecutive (non-overlapping) chunks of text with the same set of annotations:
# // Output for the example input:
# // [0, 3) / “som” -> X
# // [3, 4) / “e”-> X,Z
# // [4, 5) / “ “ -> Z
# // [5, 6) / “t” -> Z, Y
# // [6, 8) / “ex” -> Y
from typing import List, Tuple

input_data1 = [((0, 4), "some", "X"), ((3, 6), "e t", "Z"), ((5, 8), "tex", "Y")]
input_word1 = "some text"


def merge_intervals(
    word: str, input_data: List[Tuple[Tuple[int, int], str, str]]
) -> List[Tuple[Tuple[int, int], str, List[str]]]:
    input_data.sort()
    result = []
    events = []
    START = True
    END = False

    for interval, _, tag in input_data:
        events.append((interval[0], START, tag))
        events.append((interval[1], END, tag))

    events.sort()
    prev_time = events[0][0]
    tags_set = set()
    for time, flag, tag in events:
        if time != prev_time:
            new_interval = (prev_time, time)
            result.append((new_interval, word[prev_time:time], sorted(tags_set.copy())))
            prev_time = time
        if flag:
            tags_set.add(tag)
        else:
            tags_set.remove(tag)

    return result


print(merge_intervals(input_word1, input_data1))
