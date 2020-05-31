# Dependencies
import numpy as np
import io
from mydict import dictionary
from Tree_Node import Node, BinaryTree
import random
import pandas as pd
from pprint import pprint
import pydot
import os
# from PIL import Image
import graphviz
from graphviz import Source
Tree_plot = graphviz.Digraph('Tree',format='png')

G = pydot.Dot(graph_type="digraph")


# first, this function is used to make sure that the data after splits will be pure to continue with the next step in
# the decision treee algorithm
def CheckPurity(data):
    label_column = data[:, -1]
    unique_classific = np.unique(label_column)
    if len(unique_classific) == 1:
        return True
    else:
        return False

#this function is used to work on data classification where it classify data base on
# majority class
def DataClassificationFunction(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    # return positive or negative based on majority
    classific = unique_classes[index]

    return classific


# this function is a part of the splitting step that aids in the decision tree algorithm
def Split_Function(data, split_column, split_value):
    split_column_values = data[:, split_column]

    data_equal = data[split_column_values == split_value]
    data_notequal = data[split_column_values != split_value]

    return data_equal, data_notequal

# this function is a part of the splitting step that aids in the decision tree algorithm
def Potential_splits(data):
    splits = dictionary()
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        values = data[:, column_index]
        unique_values = np.unique(values)
        splits.add(column_index, unique_values)

    return splits

# To calculate information gain we need to claculate entropy, there are several other ways to get information gain as gini index
# and some other methods
def GetEntropy(data):
    # sum of prob of all classes * -log(prob)
    rate_col = data[:, -1]
    # calculate the number of each class in the given data after that we have to calc the probab by dividing by the sum of two classes
    _, num = np.unique(rate_col, return_counts=True)
    prob = num / num.sum()  # we can add counts[0] + count[1] but this may give error if a class isn't found at all
    entropy = sum(prob * -np.log2(prob))
    return entropy

#this also use the above function
def GetOverallEntropy(data_equal, data_not_equal):
    data_point = len(data_equal) + len(data_not_equal)
    p_data_equal = len(data_equal) / data_point
    p_data_not_equal = len(data_not_equal) / data_point
    overall_entropy = (p_data_equal * GetEntropy(data_equal) + p_data_not_equal * GetEntropy(data_not_equal))

    return overall_entropy

# this also aids in the step of information gain step
def Estimate_Best_Split(data, potential_splits):
    overall_entropy = 10000
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        
        for value in potential_splits.get(column_index):
            data_equal, data_not_equal = Split_Function(data, split_column=column_index, split_value=value)
            current_entropy = GetOverallEntropy(data_equal, data_not_equal)

            if current_entropy <= overall_entropy:
                overall_entropy = current_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value



# this is our main algorithm modified to use tree nodes
def DecisionTreeAlgorithmWithNodes(df, current_node, counter=0, min_samples=2, max_depth=5):
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS

        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

        # base cases
        # len of data returns the total number of features
        # upon reaching 1 feature i will break of the loop
    if (CheckPurity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = DataClassificationFunction(data)
        current_node = Node(classification)
        return current_node


    # recursive part
    else:
        counter = counter + 1

        # helper functions
        splits = Potential_splits(data)
        # get potential splits return a dicitonar of columns with unique values in each column
        if splits.counter == 0:
            classification = DataClassificationFunction(data)
            current_node = Node(classification)
            print(current_node.value)
            return current_node

        split_column, split_value = Estimate_Best_Split(data,splits)
        data_equal, data_not_equal = Split_Function(data, split_column, split_value)

        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        if counter == 1:
            current_node = Node(feature_name)

        # find answers (recursion)

        if current_node is None:
            current_node = Node(feature_name)
        current_node.right = DecisionTreeAlgorithmWithNodes(data_equal, current_node.right, counter, min_samples,
                                                            max_depth)
        current_node.left = DecisionTreeAlgorithmWithNodes(data_not_equal, current_node.left, counter, min_samples,
                                                           max_depth)

        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base cases).
        if current_node.right == current_node.left:
            current_node.value = current_node.right.value

        # else:
        #     TreeOfNodes.insert(yes_node,'yes')
        #     TreeOfNodes.insert(no_node, 'no')

        return current_node




first_time = 1
my_path = []
#classify without drawing
#this also use tree nodes in its implementation
def ClassifyExampleWithNodes(example, tree):

    if tree.left == None and tree.right == None:
        #base case of classification +ve or -Ve
        my_path.append(tree.value)
        return tree.value
        
    else:
        #it's a question 
        if example[tree.value] == 1: #a no answer
            my_path.append(tree.value)
            return ClassifyExampleWithNodes(example, tree.right)
        
        elif example[tree.value]==0: #a yes answer
            my_path.append(tree.value)
            return ClassifyExampleWithNodes(example, tree.left)


def CalculateAccuracy(df, tree):
    #Tasnim commented this
    #df["classification"] = df.apply(classify_example_with_Nodes, axis=1, args=(tree, 1,))
    #applying the classification function on the given df
    df["classification"] = df.apply(ClassifyExampleWithNodes, axis=1, args=(tree,))
    #writing the result of classification in a file
    df["classification"].to_csv('classify.csv', encoding='utf-8')
    #check if we want to calculate the accuracy or not (just printing in a file) by checking if the label column exists or not
    if 'label' in df:
        df["classification_correct"] = df["classification"] == df["label"]
        accuracy = df["classification_correct"].mean()
        return accuracy
# end of code
