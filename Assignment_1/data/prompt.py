# -*- coding:utf-8 -*-

PROMPT = """\
Generate a fluent sentence to connect two entities using their relations based on commonsense knowledge, \
but don't add extra information. (direction: from Entity 1 to Entity 2). Here are three examples:

Example 1:
Input: Entity 1 is "apple", Entity 2 is "eat", and the relation is "ReceivesAction".
Output: Apples can be eaten.

Example 2:
Input: Entity 1 is "trash", Entity 2 is "waste", and the relation is "RelatedTo".
Output: Trash is related to waste.

Example 3:
Input: Entity 1 is "fish", Entity 2 is "catch", and the relation is "HasSubevent".
Output: Fishing has a subevent of catching fish.

Task:
Input: Entity 1 is "{start_word}", Entity 2 is "{end_word}", and the relation is "{relation}".
Output: \
"""
