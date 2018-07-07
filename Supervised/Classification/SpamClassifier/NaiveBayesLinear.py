"""
Date: July 5, 2018
Author: Eric Cotner

Script for building, training, and running inference on a Naive Bayes classifier for classifying spam.
"""

# Import necessary modules
from utils import importData, formatAllMessages, makeWordMappingFromFreqList
from DataExploration import wordFrequency
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_PATH = "./Data/spam.csv"

# Define function for tokenizing message strings
def tokenizeMessages(word_map, df):
    """
    Takes in a map of words to integers, then turns all messages into numpy arrays of integers. Does this in place.
    Arguments:
          word_map: a dictionary from words to integers
          df: DataFrame containing messages in the form of numpy arrays
    Returns:
          None
    """
    # Iterate through each message in DataFrame
    for row, msg in enumerate(df["Message"]):
        # Split message into words
        msg = msg.split()
        # Create list of indices from word->index mapping; map words not in dict to 0
        msg = np.asarray([word_map.get(word, 0) for word in msg])
        # Replace message in DataFrame with tokenized list
        df.at[row, "Message"] = msg


# Define function for converting tokenized string to vector
def vectorizeTokenizedMessage(msg, word_map):
    """
    Converts tokenized message into a vector.

    :param msg: list of integer indices
    :return vec: a vector denoting the presence of a tokenized word
    """
    # Initialize empty vector
    vec = np.zeros(len(word_map)+1, dtype=float)
    # Iterate over tokens in msg
    for token in msg:
        # Flip bits in vector indicating presence of token
        vec[token] = True
    return vec


def balanceDataset(class1, class2):
    """
    Automatically balances the dataset so that there is an equal number of both classes.

    Arguments:
        class1: DataFrame containing the first class
        class2: DataFrame containing the second class
    Returns:
        X: DataFrame containing the balanced dataset
    """
    data = [class1, class2]
    # Calculate number of examples in classes, minority class, and how much of the minority class to oversample
    min_idx = np.argmin([len(class1), len(class2)])
    maj_idx = np.argmax([len(class1), len(class2)])
    n_min, n_maj = (len(data[min_idx]), len(data[maj_idx]))
    n_oversample = n_maj/n_min
    # Add majority class to X
    X = data[maj_idx].copy()
    # Add oversampled minority class to X
    for n in range(int(n_oversample)):
        X = X.append(data[min_idx]) #
    frac_n_oversample = int((n_oversample%1) * n_min)
    X = X.append(data[min_idx].iloc[:frac_n_oversample])
    # Shuffle dataset
    X = X.sample(frac=1, random_state=0).reset_index(drop=True)
    return X

def getBatches(df, batch_size, n_features):
    """
    This is a generator that yields minibatches (x,y) for the training process to iterate over by taking in a DataFrame
    and returning a tuple which encodes a sparse array representing the minibatch data.

    :param df: a DataFrame with rows "y" (integer) and "Message" (list of integers)
    :param size: the minibatch size
    :param n_features: an integer specifying the number of features
    :return: (x,y) minibatch, where x is a tuple of (indices, values, shape), and y is the label (0/1)
    """
    # Calculate number of batches
    n_batches = len(df)//batch_size
    # Return
    for b in range(n_batches):
        indices = []
        for n, msg in enumerate(df["Message"].iloc[b*batch_size:(b+1)*batch_size]):
            for idx in msg:
                indices.append([n,idx])
        values = np.ones(len(indices))
        shape = [batch_size, n_features]
        y = np.expand_dims(df["y"].iloc[b*batch_size:(b+1)*batch_size].values, axis=-1)
        yield ((np.array(indices), values, shape), y)
    indices = []
    for n, msg in enumerate(df["Message"].iloc[(b+1)*batch_size:]):
        for idx in msg:
            indices.append([n,idx])
    if (b+1)*batch_size < len(df):
        values = np.ones(len(indices))
        y = np.expand_dims(df["y"].iloc[(b+1)*batch_size:].values, axis=-1)
        shape = [len(y), n_features]
        yield ((np.array(indices), values, shape), y)


# Define the Model class
class Model(object):
    # Initialize empty untrained model
    def __init__(self, n_features, n_classes):
        self.graph = tf.Graph()
        self.n_features = n_features
        self.n_classes = n_classes
        with self.graph.as_default():
            self.X = tf.sparse_placeholder(dtype=tf.float32, shape=[None, n_features], name="Input")
            #self.W = tf.get_variable(name="W", shape=[n_features, n_classes], dtype=tf.float32,
            #                    initializer=tf.contrib.layers.xavier_initializer())
            self.W = tf.Variable(initial_value=np.random.randn(n_features, n_classes), dtype=tf.float32, name="W")
            self.b = tf.Variable(initial_value=np.zeros(shape=[n_classes], dtype=np.float32), dtype=tf.float32, name="b")
            self.Y = tf.add(tf.sparse_tensor_dense_matmul(self.X, self.W), self.b, name="Y")
            self.Y_label = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name="Y_label")
            self.L2_param = tf.placeholder(dtype=tf.float32, shape=[], name="L2_parameter")
            L2_loss = self.L2_param * tf.nn.l2_loss(self.W, name="L2_loss")
            self.loss = tf.add(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y_label, logits=self.Y)), L2_loss, name="Loss")

            tf.summary.scalar(tensor=tf.log(self.loss), name="log_loss")
            tf.summary.histogram(values=self.W, name="W_histogram")


    # Define Monitor class for monitoring progress during training
        # Initialize class
        # Define method for updating state of monitor
        # Define method for outputting data to file

    # Define training routine
    def train(self, X_train, X_val, epochs, learning_rate, batch_size, L2_parameter):
        """
        Perform training routine. X_train and X_val are in the form of pandas DataFrames composed of lists of integers.
        We need to convert these lists into vectors, then concatenate them into batches for training.
        """
        # Define some useful constants?
        # Set default graph
        with self.graph.as_default():
            # TODO: Define training optimizer and training operation
            optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam")
            train_op = optimizer.minimize(self.loss)
            # Begin tensorflow session
            with tf.Session() as sess:
                # Initialize variables, tensorboard writer, step counter
                sess.run(tf.global_variables_initializer())
                summary_merge = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(logdir="./logs/1/train", graph=self.graph)
                val_writer = tf.summary.FileWriter(logdir="./logs/1/validation", graph=self.graph)
                global_step = 0

                # Iterate over epochs
                for e in range(epochs):
                    # Iterate over minibatches
                    b = 0
                    for x, y in getBatches(X_train, batch_size, self.n_features):
                        # Run session
                        feed_dict = {self.X: x, self.Y_label: y, self.L2_param: L2_parameter}
                        _ = sess.run([train_op], feed_dict=feed_dict)
                        if (b%10==0):
                            summary = sess.run(summary_merge, feed_dict=feed_dict)
                            train_writer.add_summary(summary, global_step)
                            x, y = list(getBatches(X_val, len(X_val), self.n_features))[0]
                            feed_dict = {self.X: x, self.Y_label: y, self.L2_param: L2_parameter}
                            summary = sess.run(summary_merge, feed_dict=feed_dict)
                            val_writer.add_summary(summary, global_step)
                        print(global_step)
                        global_step += 1
                        b += 1

    # Define inference/prediction routine

# Run the script
DEBUGGING = False
if __name__ == "__main__":
    if DEBUGGING:
        # Import raw data
        df = importData(DATA_PATH)
        # Convert data into usable format by tokenizing
        formatAllMessages(df)
        word_freq = wordFrequency(df)
        word_map = makeWordMappingFromFreqList(word_freq, drop_below=2)
        tokenizeMessages(word_map, df)
        # Split into train, validation, and test sets
        spam = df[df["y"] == 1]
        ham = df[df["y"] == 0]
        print(len(spam), "+", len(ham), "=", len(spam)+len(ham))
        X_train = balanceDataset(spam.iloc[100:], ham.iloc[100:])
    else:
        # Import raw data
        df = importData(DATA_PATH)
        # Convert data into usable format by tokenizing
        formatAllMessages(df)
        word_freq = wordFrequency(df)
        word_map = makeWordMappingFromFreqList(word_freq, drop_below=2)
        tokenizeMessages(word_map, df)
        # Split into train, validation, and test sets
        spam = df[df["y"] == 1]
        ham = df[df["y"] == 0]
        X_val = pd.concat([spam.iloc[:50], ham.iloc[:50]], axis=0)
        X_test = pd.concat([spam.iloc[50:100], ham.iloc[50:100]])
        # Add multiples of minority class (spam) to training set to balance classes
        X_train = balanceDataset(spam.iloc[100:], ham.iloc[100:])

        # Instantiate model
        model = Model(n_features=(len(word_map)+1), n_classes=1)
        # Call training method
        model.train(X_train, X_val, epochs=100, learning_rate=0.001, batch_size=10, L2_parameter=0.01)
        # Save model to file


        # Run inference (do this later, after successful training)