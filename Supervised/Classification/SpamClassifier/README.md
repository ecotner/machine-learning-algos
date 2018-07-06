# Spam classifiers
This is a simple spam classifier. I think I'll try out a number of different
techniques, but for now I'll keep it simple.

## 0) Data exploration
The very first order of business is to get some data and take a look at it. I'll
be using SMS messages gathered from [Grumbletext](grumbletext.co.uk) (looks like
the website may have folded) and the NUS SMS Corpus. I gathered the data directly
from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset/home), but
it was originally taken from [this website](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/).

If we run the `DataExploration.py` script, we load the data directly from a the
`spam.csv` file, and do a little bit of formatting: strip the messages of all
punctuation, convert everything to lowercase, and then convert all numbers to the
token "000". The total number of unique words in the corpus is 8760, though this is likely
inflated as there are tons of words that are misspelled which are being counted
multiple times; the number of unique words which appear more than once is only 4053.

We can see that some of the most frequent words are really common ones such as "000"
(any instance of a number), 'to', 'i' 'you', 'a', 'the', etc. Since this is a corpus
of texts, of course the pronoun 'u' is also very prevalent.

## 1) Naive Bayes classifier
For this first attempt we'll start with a Naive Bayes classifier. Our goal is
that given some feature vector x, we can decompose the probability of 
classification y as such:

![](http://quicklatex.com/cache3/5a/ql_ac07d90ac10fd03de878f95cf7ac575a_l3.png)

Where the "naive" part comes is is that we make the assumption that all features
are conditionally independent of each other, so that P(x1|x2,y) = P(x1|y), and so
on. Therefore, the classification probability is reduced to

![](http://quicklatex.com/cache3/ca/ql_faba38a4f7f33b00059e579a21f88eca_l3.png)

which is much simpler to compute. In the simplest approximation, the features
x_i will simply be the frequency of words as they appear in a message, plus some
other features, like the number of misspelled words (or perhaps fraction of the
words that are misspelled). For the spam classification task specifically, there
are only two classes (spam and not spam), so our classification label will be
y=1 for spam and y=0 for not spam.
