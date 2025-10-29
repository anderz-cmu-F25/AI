"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""

import math
from collections import defaultdict, Counter
from math import log
# from tqdm import tqdm


def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    for tag in prev_prob:
        if i == 0:
            log_prob[tag] = prev_prob[tag] + log(emit_prob[tag][word])
            predict_tag_seq[tag] = [tag]
        else:
            max_prob = prev_prob[tag] + log(trans_prob[tag][tag]) + (log(emit_prob[tag][word]) if emit_prob[tag][word] != 0 else log(emit_prob[tag]["UNSEEN"]))
            max_tag = tag
            for tag1 in prev_prob:
                curr_prob = prev_prob[tag1] + log(trans_prob[tag1][tag]) + (log(emit_prob[tag][word]) if emit_prob[tag][word] != 0 else log(emit_prob[tag]["UNSEEN"]))
                if curr_prob > max_prob:
                    max_prob = curr_prob
                    max_tag = tag1
            log_prob[tag] = max_prob
            predict_tag_seq[tag] = prev_predict_tag_seq[max_tag] + [tag]

    return log_prob, predict_tag_seq

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    laplace = 1e-5
    word_dic = defaultdict(lambda: 0)
    tag_dic = defaultdict(lambda: 0)
    hapax = defaultdict(lambda: 0)

    for sentence in train:
        word, prev = sentence[0]
        emit_prob[prev][word] += 1
        word_dic[word] += 1
        tag_dic[prev] += 1
        for i in range(1, len(sentence)):
            word, tag = sentence[i]
            trans_prob[prev][tag] += 1
            emit_prob[tag][word] += 1
            prev = tag
            word_dic[word] += 1
            tag_dic[tag] += 1

    word_dic["UNSEEN"] = 1

    for tag in emit_prob:
        for word in emit_prob[tag]:
            if emit_prob[tag][word] == 1:
                hapax[tag] += 1
        if hapax[tag] == 0:
            hapax[tag] = 1

    hapax_sum = sum(hapax.values())
    for tag in tag_dic:
        hapax[tag] /= hapax_sum

    # print(hapax)
    # print(sum(hapax.values()))
    # exit(0)

    for tag in tag_dic:

        ttl = len(train)
        denominator = (ttl + laplace*(len(tag_dic) + 1))
        if tag == "START":
            init_prob[tag] = (ttl + laplace) / denominator
        else:
            init_prob[tag] = laplace / denominator

        ttl = sum(trans_prob[tag].values())
        denominator = (ttl + laplace*(len(tag_dic) + 1))
        for tag2 in tag_dic:
            cnt = trans_prob[tag][tag2]
            if cnt == 0:
                trans_prob[tag][tag2] = laplace / denominator
            else:
                trans_prob[tag][tag2] = (cnt + laplace) / denominator

        ttl = sum(emit_prob[tag].values())
        denominator = (ttl + laplace*(len(word_dic) + 1))
        for word in word_dic:
            cnt = emit_prob[tag][word]
            emit_prob[tag][word] = (cnt + laplace * hapax[tag]) / denominator
    
    predicts = []
    
    for sen in range(len(test)):
    # for sen in tqdm(range(len(test))):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(1e-5)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        predict = predict_tag_seq[max(log_prob, key=log_prob.get)]
        for i in range(len(sentence)):
            predict[i] = (sentence[i], predict[i])
        predicts.append(predict)
        
    return predicts