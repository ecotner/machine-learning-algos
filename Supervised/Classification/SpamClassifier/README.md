# Spam classifier
This is a simple spam classifier. I think I'll try out a number of different
techniques, but for now I'll keep it simple.

## 1) Naive Bayes classifier.
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
words that are misspelled)
