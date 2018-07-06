"""
Utility file to hold some
"""

import pandas as pd
import re

""" FUNCTIONS FOR HANDLING DATA """

def importData(path):
    """ Imports data from a csv found at <path>, does a little bit of formatting. Returns a DataFrame. """
    df = pd.read_csv(path, encoding="latin1")
    df = df[["v1","v2"]]
    df.columns = ["y", "Message"]
    df["y"] = pd.Series([0 if (e == "ham") else 1 for e in df["y"]])
    return df

""" NLP CONVENIENCE FUNCTIONS"""

def tokenizeNumbers(string):
    """ Turns all instances of numbers into "000". Returns a string. """
    pattern = re.compile(r"\d+")
    string = re.sub(pattern, "000", string)
    return string

def removePunctuation(string):
    """ Removes all punctuation; only allows alphanumeric characters. Assumes everything is already lowercase.
    Returns a string. """
    pattern = re.compile(r"[^a-z0-9\s]")
    string = re.sub(pattern, "", string)
    return string

def formatString(string):
    """ Removes all punctuation, converts all numbers to "000", and makes everything lowercase. Returns a string. """
    string = string.lower()
    string = removePunctuation(string)
    string = tokenizeNumbers(string)
    return string

def formatAllMessages(df):
    """ Formats the messages in the entire dataframe according to the formatString() function. Does this IN PLACE. """
    for msg_num, msg in enumerate(df["Message"]):
        msg = formatString(msg)
        # Replace message with formatted one
        df.at[msg_num, "Message"] = msg

def makeWordMapping(df):
    """ Generates a dictionary which maps words to a numerical index. Returns a dictionary. """
    word_mapping_dict = {}
    word_idx = 0

    # Iterate over all messages in corpus
    for msg in df["Message"]:
        # Split message into words
        msg = msg.split()
        # Iterate over words
        for word in msg:
            # Check if word is in dictionary; if not, add to dictionary and increment numerical index
            if word not in word_mapping_dict:
                word_mapping_dict[word] = word_idx
                word_idx += 1
    return word_mapping_dict

""" UNIT TESTS """

def formattingTest():
    """ Tests to see if everything is formatted properly """
    test_msgs = ["I bought like 1,000,000 donuts today!",
                 "There were over 9000 penises",
                 "Pi is approximately 3.14159",
                 "There were 5 more apples than the 20 we expected.",
                 "Just 1"]

    answer_msgs = ["i bought like 000 donuts today",
                   "there were over 000 penises",
                   "pi is approximately 000",
                   "there were 000 more apples than the 000 we expected",
                   "just 000"]
    for msg, answer in zip(test_msgs, answer_msgs):
        msg = formatString(msg)
        print("{}: {}".format(msg == answer, tokenizeNumbers(msg)))

if __name__ == "__main__":
    pass
