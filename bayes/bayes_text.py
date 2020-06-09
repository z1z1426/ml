import random

import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def word_library(document_list):
    vocab = set([])
    for i in document_list:
        vocab = vocab | set(i)
    return list(vocab)


def word2vec(vocab_list, document):
    vec_arr = np.zeros(len(vocab_list))
    for word in document:
        if word in vocab_list:
            vec_arr[vocab_list.index(word)] += 1.0
        else:
            print(word, '不在词典中！')
    return vec_arr


def bag_word2vec(vocab_list, document):
    vec_arr = np.zeros(len(vocab_list))
    for word in document:
        if word in vocab_list:
            vec_arr[vocab_list.index(word)] += 1.0
        else:
            print(word, '不在词典中！')
    return vec_arr


def bag_word2vec_list(vocab_list, document_list):
    vec_arr = np.zeros((len(document_list), len(vocab_list)))
    for i, document in enumerate(document_list):
        for j, word in enumerate(document):
            if word in vocab_list:
                vec_arr[i][vocab_list.index(word)] += 1.0
            else:
                print(word, '不在词典中！')
    return vec_arr


def train(train_mat, class_list):
    num_doc = len(train_mat)
    num_words = len(train_mat[0])
    p_ab = sum(class_list) / float(num_doc)
    p0, p1 = np.ones(num_words), np.ones(num_words)
    p0_total, p1_total = 2.0, 2.0
    for i in range(num_doc):
        if class_list[i]:
            p1 += train_mat[i]
            p1_total += np.sum(train_mat[i])
        else:
            p0 += train_mat[i]
            p0_total += np.sum(train_mat[i])
    p1_vec, p0_vec = np.log(p1 / p1_total), np.log(p0 / p0_total)
    return p0_vec, p1_vec, p_ab


def classify(doc_vec, p0_vec, p1_vec, p_ab):
    p0_chance = np.sum(doc_vec * p0_vec) + np.log(1.0 - p_ab)
    p1_chance = np.sum(doc_vec * p1_vec) + np.log(p_ab)
    return 0 if p0_chance > p1_chance else 1


def test(test_list):
    doc_list, class_list = loadDataSet()
    vocab = word_library(doc_list)
    train_vec = word2vec(vocab, doc_list)
    p0_vec, p1_vec, p_ab = train(train_vec, class_list)
    test_vec = word2vec(vocab, test_list)
    result = []
    for doc_mat in test_vec:
        result.append(classify(doc_mat, p0_vec, p1_vec, p_ab))
    return result


def textparse(bigstring):
    import re
    list_of_tokens = re.split(r'\W*', bigstring)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spamtest():
    doc_list, class_list, full_text = [], [], []
    for i in range(1, 26):
        with open('email/spam/%d.txt' % i, encoding='ISO-8859-1') as f:
            word_list = textparse(f.read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        with open('email/ham/%d.txt' % i, encoding='ISO-8859-1') as f:
            word_list = textparse(f.read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = word_library(doc_list)
    test_set, test_classes = [], []
    for i in range(10):
        rand_index = int(random.uniform(0, len(doc_list)))
        test_set.append(doc_list.pop(rand_index))
        test_classes.append(class_list.pop(rand_index))
    train_mat = bag_word2vec_list(vocab_list, doc_list)
    p0_vec, p1_vec, p_ab = train(train_mat, class_list)
    error_count = 0
    for i, test_doc in enumerate(test_set):
        word_vec = bag_word2vec(vocab_list, test_doc)
        if classify(word_vec, p0_vec, p1_vec, p_ab) != test_classes[i]:
            error_count += 1
    print('错误率为', float(error_count) / len(test_set))


def calc_most_freq(vocab_list, full_text):
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_freq[:30]


def local_words(feed1, feed0):
    import feedparser
    doc_list, class_list, full_text = [], [], []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = textparse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = textparse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = word_library(doc_list)
    top30words = calc_most_freq(vocab_list, full_text)
    for pair in top30words:
        if pair[0] in vocab_list:
            vocab_list.remove(pair[0])
    test_set, test_classes = [], []
    for i in range(20):
        rand_index = int(random.uniform(0, len(doc_list)))
        test_set.append(doc_list.pop(rand_index))
        test_classes.append(class_list.pop(rand_index))
    train_mat = bag_word2vec_list(vocab_list, doc_list)
    p0_vec, p1_vec, p_ab = train(train_mat, class_list)
    error_count = 0
    for i, test_doc in enumerate(test_set):
        word_vec = bag_word2vec(vocab_list, test_doc)
        if classify(word_vec, p0_vec, p1_vec, p_ab) != test_classes[i]:
            error_count += 1
    print('错误率为', float(error_count) / len(test_set))
    return vocab_list, p0_vec, p1_vec


def get_top_words(game, book):
    vocab_list, p0_vec, p1_vec = local_words(game, book)
    top_game, top_book = [], []
    for i in range(len(p0_vec)):
        if p0_vec[i] > -6.0:
            top_game.append((vocab_list[i], p0_vec[i]))
        if p1_vec[i] > -6.0:
            top_book.append((vocab_list[i], p1_vec[i]))
    sortedgame = sorted(top_game, key=lambda x: x[1], reverse=True)
    print('-游戏-' * 10)
    for item in sortedgame:
        print(item[0])
    sortedbook = sorted(top_book, key=lambda x: x[1], reverse=True)
    print('-书-' * 10)
    for item in sortedbook:
        print(item[0])


def test_book_game():
    import feedparser
    game = feedparser.parse('http://www.yystv.cn/rss/feed')
    book = feedparser.parse('http://www.4sbooks.com/feed')
    local_words(game, book)
    get_top_words(game, book)


if __name__ == '__main__':
    spamtest()