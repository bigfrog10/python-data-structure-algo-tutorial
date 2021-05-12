# Amazon Onsite
# Implement or Write a program that can perform file search based on file extension regex matching
# like .xml, or search files with size greater than 5MB, or other filters.
# How would you change the class to allow search conditions with AND and OR?

# https://leetcode.com/discuss/interview-question/369272/Amazon-or-Onsite-or-Linux-Find-Command
# https://leetcode.com/discuss/interview-question/799556/Amazon-Onsite
import os
import glob


def file_search(target_folder, path_filter='', attr_filter=None):
    ret = []
    for f in glob.glob(target_folder + '/**/' + path_filter, recursive=True):
        stats = os.stat(f) # mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime
        # these variables are known to outside, they can set
        fa = {'size': stats[6], 'atime': stats[7], 'mtime': stats[8], 'ctime': stats[9]}

        if attr_filter:
            if attr_filter(fa):
                ret.append(f)
        else:
            ret.append(f)

    return ret


# print(file_search('.', '*.py'))  # current folder
# print(file_search('../../..', '*.py'))
# print(file_search('../leet_code', 'q000*.py'))
# print(file_search('../leet_code', 'q000[23]*.py'))  # only q0002* and q0003* printed
# print(file_search('../leet_code', 'q009?.py'))  # match single char
print(file_search('../leet_code', 'q*.py', lambda f: f['size'] > 1000))
