import PySimpleGUI as sg
from IntegratedCode import calculate_accuracy, decision_tree_algorithm_with_nodes, my_path
from Tree_Node import Node, BinaryTree
import numpy as np
import pandas as pd
from mydict import dictionary
import csv
from datetime import datetime
from StringFilter import String_filter
import sys
import subprocess
import graphviz
Tree_plot = graphviz.Digraph('Tree',format='png')
# Column layout
sg.theme('LightGreen3')
Review_text = ""
Train_path = ""
Test_path = ""
column1 = [[sg.Button('Write a review', font='Arial', size=(45, None), button_color=('white', '#3f3f44'), )],
           ]
column2 = [[sg.Text(' ' * 30, size=(None, 1))],
           [sg.FileBrowse('Select train data', font='Arial', button_color=('white', '#3f3f44'), size=(16, None),
                          file_types=(("text files", ".csv"), ("all files", "*.*"),))],
           [sg.Text(' ' * 30, size=(None, 1))],
           [sg.FileBrowse('Select test data', font='Arial', button_color=('white', '#3f3f44'), size=(16, None),
                          file_types=(("text files", ".csv"), ("all files", "*.*"),))],
           [sg.Text(' ' * 30, size=(None, 1))],
           [sg.Button('Show accuracy', font='Arial', size=(16, None), button_color=('white', '#3f3f44'), )],
           [sg.Text(' ' * 30, size=(None, 1))],
           [sg.Button('Classify', font='Arial', size=(16, None), button_color=('white', '#3f3f44'), )],
            [sg.Text(' ' * 15, size=(None, 1))],
           [sg.Text(' ' * 5, size=(None, 1)),sg.Combo(['1', '2','3','4','5','6','7','8'],key='Depth'),sg.Text('Depth')]]
column3 = [[sg.Multiline('User log data will be displayed here :\n',text_color='#23E000', size=(45, 20), key='-OUTPUT-' + sg.WRITE_ONLY_KEY)],
           ]
column4 = [[sg.Button('Show Tree', font='Arial', size=(16, None), button_color=('black', '#fdcb9e'))],
           ]
column5 = [[sg.Button('Quit', font='Arial', size=(16, None), button_color=('black', '#fdcb9e'))],

           ]

layout = [[sg.Column(column1, justification='center')], [sg.Column(column2, justification='l'), sg.Column(column3)],
          [sg.Text(' ' * 10), sg.Column(column4, element_justification='left'), sg.Column(column5)],

          ]

# Display the window and get values

window = sg.Window('Decision Tree', layout, text_justification='left')

review_flag = 0
first_row = ['contains_No', 'contains_Please', 'contains_Thank', 'contains_apologize', 'contains_bad', 'contains_clean', 'contains_comfortable', 'contains_dirty', 'contains_enjoyed', 'contains_friendly', 'contains_glad', 'contains_good', 'contains_great', 'contains_happy', 'contains_hot', 'contains_issues', 'contains_nice', 'contains_noise', 'contains_old', 'contains_poor', 'contains_right', 'contains_small', 'contains_smell', 'contains_sorry', 'contains_wonderful']
with open('input.csv', 'w+', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(first_row)
