#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Jinci Chen
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from utils import load_data, save_data


def filter_stopwods(stop_words, words):
    """
    Input: a list of stop words, a list of words
    Output: a list of words that filtered stop words
    """
    filtered_words = [w for w in words if not w in stop_words]
    return filtered_words

def filter_punc(words):
    """
    Input: a list of words
    Output: a list of words the filter punctuation
    """
    filtered_words = [w for w in words if re.search('[A-Za-z]', w)]
    return filtered_words

if __name__ == '__main__':
    content = load_data('data/news.txt')
    stop_words = set(stopwords.words('english'))

    corpus = []
    for line in content:
        line = line.lower()
        words = word_tokenize(line)
        words = filter_stopwods(stop_words, words)
        words = filter_punc(words)
        corpus.append(' '.join(words))
    save_data(corpus, 'data/corpus.txt')

