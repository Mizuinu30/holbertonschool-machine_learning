#!/usr/bin/env python3
#Define an array
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]
# Slice the array to include the first two numbers
arr1 = arr[:2]
# Slice the array to include the last five numbers
arr2 = arr[-5:]
# Slice the array to include the 2nd through 6th numbers
arr3 = arr[1:6]
# Print the arrays
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
