"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tags = {}
    words = {}

    for sentence in train:
        for word, tag in sentence:
            if word not in words:
                words[word] = {tag: 1}
            elif tag not in words[word]:
                words[word][tag] = 1
            else:
                words[word][tag] += 1

            if tag not in tags:
                tags[tag] = 1
            else:
                tags[tag] += 1
    
    res = []
    most_seen_tag = max(tags, key=tags.get)

    for sentence in test:
        tagged_sentence = []
        for word in sentence:
            if word not in words:
                word_tag_pair = (word, most_seen_tag)
            else:
                word_tag_pair = (word, max(words[word], key=words[word].get))

            tagged_sentence.append(word_tag_pair)
        res.append(tagged_sentence)

    return res
