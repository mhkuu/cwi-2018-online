# Complex Word Identification (CWI) for the web

In this repository we research complex word identification for the web, with a minimal set-up: small lists (no more than 1000 items), simple machinery (no taggers, stemmers rather), no fancy neural network stuff, etc.

## Task

The [CWI Shared Task 2018](https://sites.google.com/view/cwisharedtask2018/) consists of predicting which words could be **difficult** and which could be *easy* for a non-native speaker, e.g.:

> "Both China and the Philippines **flexed** their *muscles* on Monday."

In this repo, we focus on the **binary classification task** for the **monolingual English** and **monolingual Spanish** tracks.

We copied the official training, development and test splits of the datasets for both languages.

Run `compare.py` to see the results for various classifiers in this repository. 

We copied [some utilities and helpers](https://github.com/sheffieldnlp/cwisharedtask2018-teaching) from [the NLP group at the University of Sheffield](https://www.sheffield.ac.uk/dcs/research/groups/natural-language-processing). Thanks a lot! 

## Datasets

All the required information is provided in a column format.
A complete description is given in the [official shared task page](https://sites.google.com/view/cwisharedtask2018/datasets).
Since we are only interested in the binary classification task, we can discard the last column (i.e., gold-standard probability). 

The English dataset (training and development) is originally divided in three files, corresponding to the source they were collected from.
We joined them together in a single file for training and a single one for development.

An example of how the data could be read is given in `utils/dataset.py`.

Since we are currently only interested in single word complexity, we removed phrases consisting of multiple words from the datasets.

## Classifiers

We have implemented the following classifiers: 

### Dummy

Trains a dummy model based on the distribution of labels in the training set.

### Baseline

Trains a model based on normalized word length and amount of tokens. This baseline was copied from [the original repository](https://github.com/sheffieldnlp/cwisharedtask2018-teaching). 

### Length-based

We have another baseline which just checks on the length of the word. Word length actually corresponds nicely with complexity! 

### Frequency-based

This classifier, additional to length, checks if the stemmed word appears in the 1000 most frequently used words in a frequency list.
- For English, the frequency list is compiled from the British National Corpus. We copied the list from List 1.2 on [this website](http://ucrel.lancs.ac.uk/bncfreq/flists.html).
- For Spanish, the frequency list is compiled from the [Corpus de Referencia del Español Actual (CREA)](http://corpus.rae.es/lfrecuencias.html).

### Character n-gram-based

This classifier checks if there are infrequent bigrams in a word, or calculates the bigram probability.
The frequency list is generated from the character bigrams in the training set, through `retrieve_ngrams.py`.

### TODO

* Add more measures :-) 

## Scoring a classifier

The `report_score` function in `utils/scorer.py` is used to assess the performance of a classifier.

`report_score` expects 2 parameters. A list of gold-standard labels (i.e. from the test data), and a list of predicted labels (i.e. what you classifier predicts on the test data).

    predicted = [1, 0, 0, 1, ...]
    actual = [1, 1, 0, 1, ...]

    report_score(actual, predicted)

A standard method for evaluating a classifier is to calculate the F1 score of its predictions.
In our case, we could compute the F1 score for each class, independently.
Since we would like an overall score that considers the performance in both classes, ``report_score`` will print the macro-F1 score.
As you will notice, there is some imbalance between the classes (the number of *complex* instances is lower than that of *simple* instances).
As such, we use macro-F1 as an overall performance metric to avoid favouring the bigger class. While this is the metric that will be used to assess your classifier, you can get more detailed per-class scores calling ``report_score`` with the parameter ``detailed=True``.

### High score

For fun purposes we'll share our weekly high scores here.

    05-10-2021
    English: Length <= 7 + frequency, macro-F1: 0.694
    Spanish: Length <= 9, macro-F1: 0.634
    12-10-2021
    English: 
        1. Logistic regression, score: 0.739
        2. N-grams, probability cut-off: 1e-12, score: 0.719
        3. N-grams, probability cut-off: 1e-11, score: 0.698
    Spanish:
        1. Length <= 9, score: 0.634
        2. Length <= 8, score: 0.622
        3. N-grams, probability cut-off: 1e-13, score: 0.619
    04-11-2021
    Spanish:
        1. Length <= 8 + frequency, score: 0.651
        2. Length <= 9 + frequency, score: 0.646
        3. Length <= 9, score: 0.634 
    11-11-2021
    German:
        1. Length <= 10, score: 0.721
        2. N-grams, probability cut-off: 1e-14, score: 0.717
        3. N-grams, probability cut-off: 1e-15, score: 0.712
