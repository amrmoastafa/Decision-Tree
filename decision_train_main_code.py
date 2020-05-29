import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random
from pprint import pprint

import pydot
import os
# from PIL import Image
import graphviz
from graphviz import Source
Tree_plot = graphviz.Digraph('Tree',format='png')

G = pydot.Dot(graph_type="digraph")

#loading and preparing data
train_df = pd.read_csv("sample_train.csv")
test_df = pd.read_csv("sample_dev.csv")
test_df = test_df.drop("reviews.text", axis=1)
test_df = test_df.rename(columns={"rating":"label"})
train_df = train_df.drop("reviews.text", axis=1)
train_df = train_df.rename(columns={"rating":"label"})

#check purity
def check_purity(data):
    label_column = data[:,-1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False

#classify data
def classify_data(data):
    label_column = data[:,-1]
    unique_classes, counts_unique_classes = np.unique(label_column,return_counts=True)
    
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification

 #potential splits
 def get_potential_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):        # excluding the last column which is the label
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

#         for index in range(len(unique_values)):
#             if index != 0:
#                 current_value = unique_values[index]
#                 previous_value = unique_values[index - 1]
#                 potential_split = (current_value + previous_value) / 2
                
#                 potential_splits[column_index].append(potential_split)

        potential_splits[column_index] = unique_values
    
    return potential_splits

 #split data
 def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]
    data_below = data[split_column_values == split_value]
    data_above = data[split_column_values !=  split_value]
    
    return data_below, data_above

 #calculating entropy
 def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

 def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy


 #determinig best split
 def determine_best_split(data, potential_splits):
    
    overall_entropy = 9999
    for column_index in potential_splits:
        #print(COLUMN_HEADERS[column_index], '-', len(np.unique(data[:, column_index])))
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

 #main algorithm
 def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data)
        if len(potential_splits)==0:
            classification = classify_data(data)
            return classification
        
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        feature_name
        #node *mynode= new node(feature_name)
        #question = "{} <= {}".format(feature_name, split_value)
        question = "{} = {}".format(feature_name, split_value)
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base cases).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree


 tree = decision_tree_algorithm(train_df, max_depth=5)

 def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    # ask question
    if comparison_operator == "=":
        print(feature_name)
        print(str(example[feature_name]))
        #if example[feature_name] <= float(value):
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        print(residual_tree)
        return classify_example(example, residual_tree)

classify_example(example, tree)

def calculate_accuracy(df, tree):

    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df["label"]
    
    accuracy = df["classification_correct"].mean()
    
    return accuracy

accuracy = calculate_accuracy(test_df, tree)