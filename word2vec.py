import os
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import multiprocessing
from multiprocessing import Pool, Manager
import threading
import shutil
import numpy as np
import pandas as pd
from collections import Counter
import time


def word2vec_kmeans(corpus, path):
    MODEL_KMEANS_PATH = path
    categories = len(corpus)
    # for i in range(categories):
    #     corpus[i] = corpus[i].split()
    # print(categories)
    if not os.path.exists(MODEL_KMEANS_PATH):
        model = Word2Vec(sentences=corpus, min_count=1, workers=multiprocessing.cpu_count())
        model.save(MODEL_KMEANS_PATH)
    else:
        model = Word2Vec.load(MODEL_KMEANS_PATH)
    words = list(model.wv.vocab.keys())
    # print(len(words))
    vectors = []
    for word in words:
        vectors.append(model.wv[word])
    clf = KMeans(n_clusters=categories)
    kmeans = clf.fit(vectors)
    labels = clf.labels_
    collection = {}
    for i in range(len(words)):
        label = labels[i]
        if label in collection:
            collection[label].append(words[i])
        else:
            collection[label] = [words[i]]

    top2000_words_kmeans = []

    writer = pd.ExcelWriter('word2vec_kmeans_top2000.xlsx')
    for i in range(categories):
        # print("cate_{}, words: {}".format(i, collection[i]))
        this_cate = pd.DataFrame({'word': collection[i]})
        # print('this cate:', this_cate)
        this_cate.to_excel(writer, sheet_name='cate_' + str(i + 1), index=None)
        for word in collection[i]:
            if word not in top2000_words_kmeans:
                top2000_words_kmeans.append(word)
    top2000_words = pd.DataFrame({'word': top2000_words_kmeans})
    # print(top2000_words)
    top2000_words.to_excel(writer, sheet_name=str(categories) + ' cate', index=None)
    writer.save()
    writer.close()
    return top2000_words_kmeans


def transition_prob(model, input_word, output_word):
    """
    based on the function score_sg_pair of Word2Vec in gensim
    """
    # get the embedding vector of input word of word2vec
    input_word_vec = model.wv[input_word]
    # get the details of output word of word2vec
    """
    print(output_word): example
    Vocab(code:array([0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1],
      dtype=uint8), count:9235, index:34466, point:array([352194, 352192, 352188, 352180, 352166, 352140, 352091, 351996,
       351823, 351505, 350947, 349957, 348291, 345585, 341333, 334882,
       325262, 311061, 291240], dtype=uint32), sample_int:4294967296)
    """
    output_word = model.wv.vocab[output_word]
    # model.syn1[output_word.point]: the vectors of nodes on the huffman code/path of output word
    node_vector = model.trainables.syn1[output_word.point].T
    # matrix multiply: calculate the probability from input_word to each node of huffman path of output_word
    # dot_value: p(nodes of output_word | input_word)
    input2node_prob = np.dot(input_word_vec, node_vector)
    # calculate the log probability from input_word to output_word
    # np.logaddexp(x, y): log(exp(x)+exp(y))
    # output_word.code: the code of nodes (1 or 0)
    """
    the log probability of each node = −log(1+exp(−x_⊤* θ))−dx_⊤* θ
    for this output_word, calculate the sum of log probability of each node
    """
    input2output_prob = -sum(np.logaddexp(0, -input2node_prob) + output_word.code * input2node_prob)
    return input2output_prob


def word2vec_huffman(corpus, path):
    """
    content: text
    wi: a word in text

    if wi is a keyword in text, then wi would maximize p(content|wi)
    the larger the p(content|wi), the larger the probability of content appearing under the condition of wi

    thus, for every word in text, calculate p(content|wi) and sort in descending order
    words in the front can be considered as keywords of text
    """
    MODEL_HUFFMAN_PATH = path
    if not os.path.exists(MODEL_HUFFMAN_PATH):
        model = Word2Vec(sentences=corpus, min_count=5, workers=multiprocessing.cpu_count(), window=10, size=256, sg=1, hs=1, iter=10)
        model.save(MODEL_HUFFMAN_PATH)
    else:
        model = Word2Vec.load(MODEL_HUFFMAN_PATH)
    categories = len(corpus)
    """
    none
    """
    # top2000_words = []
    # writer = pd.ExcelWriter('word2vec_huffman_top2000_none.xlsx')
    # for i in range(categories):
    #     # extract words which in both corpus and trained model
    #     words = [word for word in corpus[i] if word in model.wv]
    #     start = time.time()
    #     # calculate p(all words| word) for each word
    #     input2output_all = {input_word: sum([transition_prob(model, input_word, output_word) for output_word in words])
    #                         for input_word in words}
    #     print(pd.Series(Counter(input2output_all).most_common(2000)))
    #     print('cate_' + str(i + 1), 'time taken:', time.time() - start, 's')
    #
    #     this_top2000_words = [word for word, value in Counter(input2output_all).most_common()[:2000]]
    #     this_top2000_values = [value for word, value in Counter(input2output_all).most_common()[:2000]]
    #     top_word_value = pd.DataFrame({'word': this_top2000_words, 'value': this_top2000_values})
    #     sheet_name = 'cate_' + str(i + 1)
    #     top_word_value.to_excel(writer, sheet_name=sheet_name, index=None)
    #     for word in this_top2000_words:
    #         if word not in top2000_words:
    #             top2000_words.append(word)
    #
    # top_words = pd.DataFrame({'word': top2000_words})
    # top_words.to_excel(writer, sheet_name=str(categories) + ' cate', index=None)
    # writer.save()
    # writer.close()

    """
    multithreading
    """
    # writer = pd.ExcelWriter('word2vec_huffman_top2000_multithreading.xlsx')
    # dict = {}
    # for i in range(categories):
    #     # extract words which in both corpus and trained model
    #     words = [word for word in corpus[i] if word in model.wv]
    #     print('start thread', str(i + 1))
    #     thread = huffman_multithreading(name='cate_' + str(i + 1), model=model, words=words)
    #     dict[i] = thread
    #     thread.start()

    """
    multiprocessing
    """
    writer = pd.ExcelWriter('word2vec_huffman_top2000_multiprocessing.xlsx')
    pool = Pool(multiprocessing.cpu_count())
    manager = Manager()
    top2000_words = manager.list()
    for i in range(categories):
        # extract words which in both corpus and trained model
        words = [word for word in corpus[i] if word in model.wv]

        pool.apply_async(huffman_multiprocessing, (model, words, i, top2000_words), callback=write2file)
        # print(this_top2000_words)
        # top_word_value = pd.DataFrame({'word': this_top2000_words, 'value': this_top2000_values})
        # temp_name = 'cate_' + str(i + 1) + '.xlsx'
        # print('cate', (i + 1), 'writing into excel ...')
        # top_word_value.to_excel(temp_name, index=None)
        # print('cate', (i + 1), 'writing into excel done')
    pool.close()
    pool.join()

    top2000_words = list(top2000_words)
    top_words = pd.DataFrame({'word': top2000_words})

    for i in range(categories):
        sheet_name = 'cate_' + str(i + 1)
        data_name = 'word2vec_huffman_temp/' + sheet_name + '.xlsx'
        data = pd.read_excel(data_name)
        data.to_excel(writer, sheet_name=sheet_name, index=None)

    top_words.to_excel(writer, sheet_name=str(categories) + ' cate', index=None)
    writer.save()
    writer.close()

    shutil.rmtree('word2vec_huffman_temp')

    return top2000_words


class huffman_multithreading(threading.Thread):

    def __init__(self, name=None, model=None, words=None):
        self.model = model
        self.words = words
        threading.Thread.__init__(self, name=name)

    def run(self):
        print(self.name, 'start running ...')
        start = time.time()
        input2output_all = {input_word: sum([transition_prob(self.model, input_word, output_word) for output_word in self.words])
                            for input_word in self.words}
        print('time taken:', time.time() - start, 's')
        print(pd.Series(Counter(input2output_all).most_common(2000)))


def huffman_multiprocessing(model, words, i, top2000_words):
    print('start processing', str(i + 1))
    start = time.time()
    input2output_all = {input_word: sum([transition_prob(model, input_word, output_word) for output_word in words])
                        for input_word in words}
    print('cate', (i + 1), 'time taken:', time.time() - start, 's')
    print(pd.Series(Counter(input2output_all).most_common(2000)))
    this_top2000_words = [word for word, value in Counter(input2output_all).most_common()[:2000]]
    this_top2000_values = [value for word, value in Counter(input2output_all).most_common()[:2000]]
    for word in this_top2000_words:
        if word not in top2000_words:
            top2000_words.append(word)
    return (this_top2000_words, this_top2000_values, i)


def write2file(res):
    this_top2000_words = res[0]
    this_top2000_values = res[1]
    i = res[2]
    top_word_value = pd.DataFrame({'word': this_top2000_words, 'value': this_top2000_values})
    if not os.path.exists('word2vec_huffman_temp'):
        os.makedirs('word2vec_huffman_temp')
    temp_name = 'word2vec_huffman_temp/cate_' + str(i + 1) + '.xlsx'
    print('cate', (i + 1), 'writing into excel ...')
    top_word_value.to_excel(temp_name, index=None)
    print('cate', (i + 1), 'writing into excel done')
    return