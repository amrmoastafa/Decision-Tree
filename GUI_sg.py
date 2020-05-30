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
while True:
    event, values = window.read()

    if event in (sg.WIN_CLOSED, 'Quit'):
        print(event, values)
        print(values)
        break
    elif event is 'Show Tree':
        imageViewerFromCommandLine = {'linux': 'xdg-open',
                                      'win32': 'explorer',
                                      'darwin': 'open'}[sys.platform]
        subprocess.run([imageViewerFromCommandLine, 'Tree.gv.png'])
        print('show')
    elif event is 'Write a review':
        Review_text = sg.popup_get_text('Enter your review')
        if Review_text is not None:
            window['-OUTPUT-' + sg.WRITE_ONLY_KEY].print("\nUser entered a review : "+Review_text, text_color='black')
            Review_text = String_filter(Review_text, 0)
            with open('input.csv', 'w+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(first_row)
            rev = int(Review_text)
            print(len(Review_text))
            arr = list(Review_text)
            print(arr)
            fl = open('input.csv', 'a+')
            writer = csv.writer(fl)
            writer.writerow(arr)
            fl.close()
            print("done")
            review_flag = 1
    elif event is 'Show accuracy':
        if list(values.values())[2] is '':
            depth = 5
        else:
            depth = int(list(values.values())[2])
        Tree_plot = graphviz.Digraph('Tree', format='png')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        window['-OUTPUT-' + sg.WRITE_ONLY_KEY].print(" ", text_color='black')
        window['-OUTPUT-' + sg.WRITE_ONLY_KEY].print("Starting the execution at : " + current_time, text_color='black')

        Train_path = list(values.values())[0]
        print("Train path : " + Train_path)
        Test_path = list(values.values())[1]
        print("Test_path :" + Test_path)

        test_df = pd.read_csv(Test_path)
        train_df = pd.read_csv(Train_path)
        train_df = train_df.drop("reviews.text", axis=1)
        train_df = train_df.rename(columns={"rating": "label"})
        test_df = test_df.drop("reviews.text", axis=1)
        test_df = test_df.rename(columns={"rating": "label"})

        TreeOfNodes = BinaryTree()
        TreeOfNodes.root = decision_tree_algorithm_with_nodes(train_df, TreeOfNodes.root, max_depth=depth)
        acc = calculate_accuracy(test_df, TreeOfNodes.root) * 100
        acc = round(acc, 3)
        print(acc)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        window['-OUTPUT-' + sg.WRITE_ONLY_KEY].print("accuracy is: " + str(acc) + '%', text_color='black')
        window['-OUTPUT-' + sg.WRITE_ONLY_KEY].print("Execution Finished At : " + current_time, text_color='black')
        def draw_tree(leftnode = TreeOfNodes.root.left, node = TreeOfNodes.root, rightnode = TreeOfNodes.root.right):

            if node.right is not None or node.left is not None:
                if node.right.value == node.left.value:
                    # print(node.right.value)
                    Tree_plot.node(node.right.value+node.value, node.right.value)
                    Tree_plot.edge(node.value, node.right.value+node.value, label="Yes or No")
                else:
                    # print('contains_'+node.value.split('contains_')[1])
                    Tree_plot.node(node.value,'contains_'+node.value.split('contains_')[1])
                    Tree_plot.node(node.left.value + node.value, node.left.value)
                    Tree_plot.node(node.right.value + node.value ,node.right.value)

                    Tree_plot.edge(node.value,node.right.value + node.value,label="Yes")
                    Tree_plot.edge(node.value,node.left.value + node.value,label= "No")
                    node.left.value = node.left.value + node.value
                    node.right.value = node.right.value + node.value
                    draw_tree(leftnode = leftnode.left, node= leftnode,rightnode = leftnode.right)
                    draw_tree(leftnode = rightnode.left, node= rightnode,rightnode = rightnode.right)

        draw_tree()
        Tree_plot.render()
