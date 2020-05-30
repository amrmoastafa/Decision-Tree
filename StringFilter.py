import pandas as pd
from mydict import dictionary
import numpy as np
import re


def String_filter(str, counter_0_1=0):
    for char in str:
        # Counting the number of 0s and 1s
        if char == '0' or char == '1':
            counter_0_1 = counter_0_1 + 1
    if len(str) is counter_0_1:
        # if the total length = the count of 0s and 1s
        if len(str) < 25:
            diff = 25 - len(str)
            for i in range(diff):
                str += '0'
            return str
        # if the total length is greater than 25 return only the first 25 0s and 1s
        elif len(str) > 25:
            return str[0:25]
    if len(str) is 25 and counter_0_1 is 25:
        return str

    else:
        # Filterting the string by first getting the column headers
        # then lowering the case of all characters and removing non alphabetical or space characters
        # comparing the string by each feature and concatenating the result
        string_returned = ''
        try:
            train_df = pd.read_csv("sample_train.csv")
            train_df = train_df.drop("reviews.text", axis=1)
            train_df = train_df.drop("rating", axis=1)
            list_of_features = list(train_df.columns)
            for i in range(len(list_of_features)):
                list_of_features[i] = list_of_features[i].lower()
                list_of_features[i] = list_of_features[i][9:] + ' '
        except:
            list_of_features = ['no ', 'please ', 'thank ', 'apologize ', 'bad ', 'clean ', 'comfortable ', 'dirty ',
                                'enjoyed ', 'friendly ', 'glad ', 'good ', 'great ', 'happy ', 'hot ', 'issues ',
                                'nice ', 'noise ', 'old ', 'poor ', 'right ', 'small ', 'smell ', 'sorry ',
                                'wonderful ']
        str = str.lower()
        str = str.replace('.', ' ')
        regex = re.sub("[^a-zA-Z ]+", "", str)
        str = regex + ' '

        print(list_of_features)
        for feature in list_of_features:
            if str.find(feature) is not -1:
                string_returned += '1'
            else:
                string_returned += '0'
        return string_returned


# Test Cases
# Normal user input
stri1 = "We loved this hotel. *location  room  service  it has it all... almost.The location is walking distance to The " \
        "Financial District  Union Square and Chinatown and a few blocks from the cable car which will take you to the " \
        "Wharf. The room  we had a suite  was spacious and comfortable. Only drawback was lack of a view (we looked " \
        "into another building... creepy) We felt like we were in a hole with no windows because we couldn't open our " \
        "curtains due to feeling like we were constantly being watched.The service was top notch... Oliver felt like a " \
        "personal friend. He was genuinely helpful and anticipated our needs before we knew we needed him. He suggested " \
        "wonderful restaurants  parking tips  sightseeing info... the list goes on and on. The front desk was pleasant " \
        "and the bartender made wonderful drinks.Loved the roof top patio... a little difficult to find but worth the " \
        "walk up the stairs from the 10th fl.We will definitely stay at the Orchard Garden the next time we visit " \
        "SF!Thank you so much for the stellar review. Although I am very pleased to read you enjoyed your stay  I " \
        "apologize the view was less than desirable. Our property was built in 2006 on a vacant lot next to some tall " \
        "neighboring buildings. I am thrilled Oliver and the front desk team we were able to make your stay more " \
        "pleasurable. I am extremely fortunate to work with such an enthusiastic team.All of us here look forward to " \
        "your next San Francisco visit.All the best Quinn McEnteeFront Office Manager "
# User input 0s and 1s that donot add to 25
stri2 = '111110000111111'

# User input a string that is less that 25 characters :(

stri3 = 'This play is nice and wonderful'

# User input a string of 0s  and 1s greater than 25

stri4 = '111111111111111111110000100000000000'

