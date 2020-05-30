import numpy as np
import pandas as pd
from mydict import dictionary
# import matplotlib.pyplot as plt
# import seaborn as sns

import random
from pprint import pprint


class Node:
    # Node class is meant to be used only by the
    # Tree class , so no more functions need to be
    # Added.
    def __init__(self, value=None):
        self.value = value
        self.right = None
        self.left = None

    def __str__(self, level=0):
        ret = "\t"*level+repr(self.value)+"\n"
        if self.right is not None:
            ret += self.right.__str__(level+1)
        if self.left is not None:
            ret += self.left.__str__(level + 1)
        return ret



    def print_content(self):
        print("Data : " + str(self.data))
        print("left : " + str(self.left))
        print("right : " + str(self.right))


class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, value, branch):
        if self.root == None:
            self.root = Node(value)
            print("Value for root : "+str(value))

        else:
            self._insert(value, branch, self.root)

    def _insert(self, value, branch, cur_node):

        if branch is 'no':
            if cur_node.left is None:
                cur_node.left = Node(value)
            else:
                self._insert(value, 'no', cur_node.left)
        elif branch is 'yes':
            if cur_node.right is None:
                cur_node.right = Node(value)

            else:
                self._insert(value, 'yes', cur_node.right)
        else:
            print
            "Value already in tree!"

    # def add_node_left(self, value, current_node):
    #     # if the tree has no root , set this
    #     # node as root , else add this node to
    #     # tree branch
    #     if self.root is None:
    #         self.root = Node(value)
    #     else:
    #         # if the value is false , and there is no
    #         # left child for the current node , then
    #         # set the given node as the current's left child
    #         # else , call the insert function and pass to it
    #         # the left child as the current node
    #         if current_node.left is None:
    #             current_node.left = Node(value)
    #         else:
    #             self.add_node(value, current_node.left)

    # def add_node_right(self, value, current_node):
    #     if self.root is None:
    #         self.root = Node(value)
    #     else:
    #         # if the value is false , and there is no
    #         # left child for the current node , then
    #         # set the given node as the current's left child
    #         # else , call the insert function and pass to it
    #         # the left child as the current node
    #         if current_node.right is None:
    #             current_node.left = Node(value)
    #         else:
    #             self.add_node(value, current_node.right)

    def height(self, current_node, current_height=0):
        if self.root is not None:
            if current_node is None:
                return current_height
            else:
                left_height = self.height(current_node.left, current_height + 1)
                right_height = self.height(current_node.right, current_height + 1)
                return max(left_height, right_height)
        else:
            return 0

    def print_tree(self, current_node):


        if current_node is not None:
            self.print_tree(current_node.left)

            print(str(current_node.value))
            self.print_tree(current_node.right)



def fill_tree(tree, num_elems=5, max_int=5):
    from random import randint
    tree.insert('wee','no')
    for x in range(num_elems):
        cur_elem = randint(0, max_int)
        tree.insert('wee', 'yes')
        tree.insert('wee', 'no')
        # tree.add_node_left(cur_elem, node)
        # tree.add_node_right(cur_elem, node)

    return tree


# tree = BinaryTree()
#
# fill_tree(tree)
# tree.print_tree(tree.root)
# print(tree.height(tree.root))
# print(tree.root.left.left.left.left.left.value)
# print(tree.root)



