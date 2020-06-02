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


def create_vocablist(dataset):
    vocab_set = set([])
    for i in dataset:
        vocab_set = vocab_set | set(i)
    return list(vocab_set)


def words2vec(vocablist, document):
    doc_vec = [0] * len(vocablist)
    for word in document:
        if word in vocablist:
            doc_vec[vocablist.index[word]] = 1
        else:
            print('%s该词不在词库里' % word)
    return doc_vec


def train_bayes(train_matrix, train_category):
    num_doc = len(train_matrix)
    num_word = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_doc)
    p0, p1 = np.ones(num_word), np.ones(num_word)
    p0_total, p1_total = 2.0, 2.0
    for i in range(num_doc):
        if train_category[i] == 1:
            p1 += train_matrix[i]
            p1_total += sum(train_matrix[i])
        else:
            p0 += train_matrix[i]
            p0_total += sum(train_matrix[i])
    p1_vec, p0_vec = p1 / p1_total, p0 / p0_total
    return p0_vec, p1_vec, p_abusive


'''
def set_of_words2vec(vocablist, input_set):
    return_vec = [0] * len(vocablist)
    for word in input_set:
        if word in vocablist:
            return_vec[vocablist.index(word)] = 1
        else:
            print('the word:%s is not in my vocabulary!' % word)
    return return_vec
'''


def train_nbo(train_matrix, train_category):
    num_traindocs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_traindocs)
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom, p1_denom = 2.0, 2.0
    for i in range(num_traindocs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = np.log(p1_num / p1_denom)
    p0_vect = np.log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2classify, p0_vec, p1_vec, p_class1):
    p1 = np.sum(vec2classify * p1_vec) + np.log(p_class1)
    p0 = np.sum(vec2classify * p0_vec) + np.log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_nb():
    list_oposts, list_classes = loadDataSet()
    my_vocablist = create_vocablist(list_oposts)
    train_mat = []
    for post_in_doc in list_oposts:
        train_mat.append(set_of_words2vec(my_vocablist, post_in_doc))
    p0_v, p1_v, p_ab = train_nbo(train_mat, list_classes)
    test_entry = ['love', 'my', 'dalmation']
    this_doc = np.array(set_of_words2vec(my_vocablist, test_entry))
    print(test_entry, 'classified as: ', classify_nb(this_doc, p0_v, p1_v, p_ab))
    test_entry = ['stupid', 'garbage']
    this_doc = np.array(set_of_words2vec(my_vocablist, test_entry))
    print(test_entry, 'classified as: ', classify_nb(this_doc, p0_v, p1_v, p_ab))


if __name__ == '__main__':
    testing_nb()
