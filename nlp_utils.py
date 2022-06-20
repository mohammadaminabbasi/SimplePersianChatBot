import numpy as np
from hazm import Stemmer, word_tokenize

stemmer = Stemmer()


def tokenize(sentence):
    return word_tokenize(sentence)


def stem(word):
    stemmer.stem(word)
    return stemmer.stem(word)


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
