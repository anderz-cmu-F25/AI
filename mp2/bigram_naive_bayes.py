# bigram_naive_bayes.py
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
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
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
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=0.5, bigram_laplace=1.0, bigram_lambda=0.5, pos_prior=0.75, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    yhats = []

    # trainning phase
    nai_pos = {}
    nai_neg = {}
    nai_vocab = set()
    bi_pos = {}
    bi_neg = {}
    bi_vocab = set()

    # Iterate through each list of words
    for i in tqdm(range(len(train_set)), desc="Trainning phase"):

        # Determine which dictionaries to use
        nai_dic = nai_pos
        bi_dic = bi_pos
        if not train_labels[i]:
            nai_dic = nai_neg
            bi_dic = bi_neg

        for j in range(len(train_set[i])):
            word = train_set[i][j]
            word_pair = (train_set[i][j], train_set[i][j+1]) if j+1 < len(train_set[i]) else ()
            nai_vocab.add(word)

            if word in nai_dic:
                nai_dic[word] += 1
            else:
                nai_dic[word] = 1
            # Only update dictionary and vocab if word pair is not empty
            if word_pair:
                bi_vocab.add(word_pair)
                if word_pair in bi_dic:
                    bi_dic[word_pair] += 1
                else:
                    bi_dic[word_pair] = 1

    nai_pos_tot = sum(nai_pos.values())
    nai_neg_tot = sum(nai_neg.values())
    bi_pos_tot = sum(bi_pos.values())
    bi_neg_tot = sum(bi_neg.values())

    # development phase
    for words in tqdm(dev_set, desc="development phase"):
        nai_pos_prob = math.log(pos_prior)
        nai_neg_prob = math.log(1-pos_prior)
        bi_pos_prob = math.log(pos_prior)
        bi_neg_prob = math.log(1-pos_prior)

        for i in range(len(words)):
            word = words[i]
            word_pair = (words[i], words[i+1]) if i+1 < len(words) else ()
            nai_pos_prob += math.log((nai_pos.get(word, 0) + unigram_laplace) / (nai_pos_tot + unigram_laplace*(len(nai_vocab)+1)))
            nai_neg_prob += math.log((nai_neg.get(word, 0) + unigram_laplace) / (nai_neg_tot + unigram_laplace*(len(nai_vocab)+1)))

            if word_pair:
                bi_pos_prob += math.log((bi_pos.get(word_pair, 0) + bigram_laplace) / (bi_pos_tot + bigram_laplace*(len(bi_vocab)+1)))
                bi_neg_prob += math.log((bi_neg.get(word_pair, 0) + bigram_laplace) / (bi_neg_tot + bigram_laplace*(len(bi_vocab)+1)))

        tot_pos = (1-bigram_lambda)*nai_pos_prob + bigram_lambda*bi_pos_prob
        tot_neg = (1-bigram_lambda)*nai_neg_prob+bigram_lambda*bi_neg_prob

        yhats.append(1 if tot_pos > tot_neg else 0)

    return yhats
