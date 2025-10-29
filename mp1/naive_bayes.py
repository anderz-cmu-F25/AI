# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.8, silently=False):
    print_values(laplace,pos_prior)
    yhats = []

    # trainning phase
    pos = {}
    neg = {}
    vocab = set()

    # Iterate through each list of words
    for i in tqdm(range(len(train_set)), desc="Trainning phase"):
        for word in train_set[i]:
            dic = pos if train_labels[i] else neg
            vocab.add(word)
            if word in dic:
                dic[word] += 1  # Increment count if word already exists
            else:
                dic[word] = 1  # Initialize count if word is new

    pos_tot = sum(pos.values())
    neg_tot = sum(neg.values())

    # development phase
    for words in tqdm(dev_set, desc="development phase"):
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1-pos_prior)
        for word in words:
            pos_prob += math.log((pos.get(word, 0) + laplace) / (pos_tot + laplace*(len(vocab)+1)))
            neg_prob += math.log((neg.get(word, 0) + laplace) / (neg_tot + laplace*(len(vocab)+1)))
        yhats.append(1 if pos_prob > neg_prob else 0)

    return yhats
