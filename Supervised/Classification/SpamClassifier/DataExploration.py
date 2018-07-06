"""
Date: July 4, 2018
Author: Eric Cotner

Data exploration script for SMS spam/ham messages
"""

from utils import importData, makeWordMapping, formatAllMessages
import numpy as np
import nltk
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "./Data/spam.csv"

def wordFrequency(df):
    """ Returns a list of tuples (word, count) sorted by decending <count>. """
    word_count = {}

    # Contruct dict of word counts
    for msg in df["Message"]:
        msg = msg.split()
        for word in msg:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
    # Sort based on frequency of word use
    word_count = [(word, word_count[word]) for word in word_count]
    word_count.sort(key=lambda tup: tup[1], reverse=True)
    return word_count

def wordHistogram(word_freq):
    """ Takes in the counts of words in the corpus and makes a histogram """
    pass

if __name__ == "__main__":

    # Import the data and format it
    df = importData(DATA_PATH)
    formatAllMessages(df)

    # Make a mapping of words to unique integer identifiers
    word_map = makeWordMapping(df)

    # Get a sorted list of the counts of various words
    word_freq = wordFrequency(df)

    print("Number of unique words: {}".format(len(word_map)))
    n_conditionally_unique_words = []
    for word, count in word_freq:
        if count > 1:
            n_conditionally_unique_words.append(word)
    print("Number of unique words ignoring words that appear once: {}".format(len(n_conditionally_unique_words)))
    print("Most frequent words:")
    for n in range(10):
        print(word_freq[n])
    print()
    print("Least frequent words:")
    for n in range(len(word_freq)-1,len(word_freq)-1-10,-1):
        print(word_freq[n])
    print()
    print("Sample of messages:")
    print(df["Message"].head())
    print()

    n_spam = len(df[df["y"] == 1])
    n_ham = len(df[df["y"] == 0])
    n_total = len(df)
    print("Number of spam: {}/{}={:.3f}".format(n_spam, n_total, n_spam/n_total))
    print("Number of non-spam: {}/{}={:.3f}".format(n_ham, n_total, n_ham/n_total))

