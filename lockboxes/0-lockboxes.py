#!/usr/bin/env python3
""" Module lockboxes"""

python

def canUnlockAll(boxes):
    """
    Determine if all boxes can be unlocked.
    
    :param boxes: List of lists representing keys inside each box.
    :return: True if all boxes can be opened, False otherwise.
    """
    n = len(boxes)
    opened = [False] * n
    opened[0] = True
    keys = set(boxes[0])
    stack = list(keys)

    while stack:
        key = stack.pop()

        if key < n and not opened[key]:
            opened[key] = True
            stack.extend(boxes[key])
    return all(opened)
