import pandas as pd
import jieba
import multiprocessing
from multiprocessing import Pool, Manager
from flask import Flask, render_template, request, send_from_directory, make_response
import time
from word2vec import word2vec_kmeans, word2vec_huffman
from chi_square import chi_square
from tf_idf import tf_idf
import os


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


DATA_PATH = 'data/1.xlsx'
STOPWORDS_PATH = 'data/stop_words_ch.txt'
DICT_PATH = 'data/new_dict.txt'
EMBEDDING_PATH = 'data/sgns.wiki.bigram-char'
MODEL_KMEANS_PATH = 'data/word2vec_model/word2vec_embedding'
MODEL_HUFFMAN_PATH = 'word2vec_model/word2vec_wx'
punc = "，。、【 】:“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥*"
dict_data = ['ETC', 'EMS', '4S店', 'GDP', 'HPV', 'IPTV', 'P2P', 'etc', '12345APP',
             '12345app', '12345热线', 'ATM', 'atm', '12123APP', '12123app']


def generateStopwords(path):
    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    return stopwords


# judge if chinese
def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def jieba_multiprocessing(line):
    cate = line[0]
    content_pre = jieba.lcut(line[1])
    line_post = []
    for word in content_pre:
        if word not in stopwords and word not in punc and is_chinese(word) and len(word) > 1:
            line_post.append(word)
    return cate, line_post


def data_processing(data_path, stopwords_path, dict_path):
    data_raw = pd.read_excel(data_path, header=None)

    jieba.load_userdict(dict_path)
    """
    uniprocessing
    """
    # print('start uniprocessing jieba')
    # data = {}
    # categories = []
    # num = len(data_raw)
    # for i in range(num):
    #
    #     category = data_raw[0][i]
    #     temp = data.get(category, [])
    #     if category not in categories:
    #         categories.append(category)
    #
    #     content = jieba.lcut(data_raw[1][i])
    #     for word in content:
    #         if word not in stopwords and word not in punc and is_chinese(word) and len(word) > 1:
    #             temp.append(word)
    #             data[category] = temp
    # print('uniprocessing jieba done')

    """
    multiprocessing
    """
    print('start multiprocessing jieba')
    cpu_num = multiprocessing.cpu_count()
    pool = Pool(cpu_num - 1)
    data_post_tokenize = pool.map(jieba_multiprocessing, zip(data_raw[0], data_raw[1]))
    pool.close()
    pool.join()

    data = {}
    categories = []
    num = len(data_post_tokenize)
    for i in range(num):

        category = data_post_tokenize[i][0]
        temp = data.get(category, [])
        if category not in categories:
            categories.append(category)

        content = data_post_tokenize[i][1]
        for word in content:
            if word not in stopwords and word not in punc and is_chinese(word) and len(word) > 1:
                temp.append(word)
                data[category] = temp
    print('multiprocessing jieba done')

    corpus = list(data.values())

    # delete words that appear in all categories many times
    categories = len(corpus)
    word_frequency = [[]]
    corpus_post = [[]]

    for i in range(categories - 1):
        word_frequency.append([])
        corpus_post.append([])
        word_frequency[i] = count(corpus[i])

    word_frequency[-1] = count(corpus[-1])

    all_words = list(word_frequency[0].keys())
    corpus_post[0] = list(word_frequency[0].keys())
    for i in range(1, categories):
        all_words.extend(word_frequency[i].keys())
        corpus_post[i] = list(word_frequency[i].keys())
        # print(corpus_post[i])

    words_deleted = []
    for i in range(len(all_words)):
        check = 0
        for j in range(categories):
            if word_frequency[j].get(all_words[i], 0) > 80:
                check += 1
        if check == categories and all_words[i] not in words_deleted:
            words_deleted.append(all_words[i])

    for word in words_deleted:
        for i in range(categories):
            corpus_post[i].remove(word)

    for i in range(categories):
        # print(corpus_post[i])
        corpus_post[i] = " ".join(corpus_post[i])
        # print(corpus_post[i])
        # print('----------------')

    # print(corpus_post)
    return corpus_post


def count(word_list):
    count = {}
    for word in word_list:
        count[word] = count.get(word, 0) + 1

    # count = sorted(list(count), key=lambda x: x[1], reverse=True)
    return count


@app.route('/upload')
def index():
    return render_template('keywords.html')


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        path = 'uploaded/' + file.filename
        file.save(path)
        global UPLOAD_DATA_PATH
        UPLOAD_DATA_PATH = path
    return render_template('keywords.html', prompt='successfully uploaded')
    # return 'successfully uploaded'


@app.route('/tokenize')
def tokenize():
    print(UPLOAD_DATA_PATH)
    global corpus
    global stopwords
    print('start processing data ...')
    start_time = time.time()
    stopwords = generateStopwords(STOPWORDS_PATH)
    corpus = data_processing(UPLOAD_DATA_PATH, STOPWORDS_PATH, DICT_PATH)
    print('jieba tokenizing cost:', time.time() - start_time, 's')
    for i in corpus:
        print(i)
    return 'successfully tokenized'


@app.route('/download/<file>')
def download(file):
    current_path = os.getcwd()
    return send_from_directory(current_path, file, as_attachment=True)


@app.route('/run_tf_idf')
def run_tf_idf():
    if not os.path.exists('tf_idf_top2000.xlsx'):
        print('start tf-idf method ...')
        print(type(corpus[0]))
        start_time = time.time()
        keywords_tf_idf = tf_idf(corpus)
        print(keywords_tf_idf)
        print('total time taken:', time.time() - start_time, 's')

    res_path = 'tf_idf_top2000.xlsx'

    return res_path


@app.route('/run_word2vec')
def run_word2vec():
    if not os.path.exists('word2vec_huffman_top2000_multiprocessing.xlsx'):
        print('start huffman method ...')
        start_time = time.time()
        this_corpus = corpus
        categories = len(this_corpus)
        if type(corpus[0]) != str:
            for i in range(categories):
                corpus[i] = " ".join(corpus[i])
        for i in range(categories):
            corpus[i] = corpus[i].split()
        keywords_huffman = word2vec_huffman(this_corpus, MODEL_HUFFMAN_PATH)
        print(keywords_huffman)
        print('total time taken:', time.time() - start_time, 's')

    res_path = 'word2vec_huffman_top2000_multiprocessing.xlsx'

    return res_path


@app.route('/run_chi_square')
def run_chi_square():
    if not os.path.exists('chi_square_top2000.xlsx'):
        print('start chi_square method ...')
        start_time = time.time()
        keywords_chi_square = chi_square(DATA_PATH, DICT_PATH, corpus)
        print(keywords_chi_square)
        print('total time taken:', time.time() - start_time, 's')

    res_path = 'chi_square_top2000.xlsx'

    return res_path


@app.route('/run_intersection')
def run_intersection():
    print('start intersection ...')
    start_time = time.time()
    if not os.path.exists('tf_idf_top2000.xlsx'):
        # tf-idf
        print('start tf-idf method ...')
        start_time = time.time()
        keywords_tf_idf = tf_idf(corpus)
        print(keywords_tf_idf)
        print('total time taken:', time.time() - start_time, 's')

    if not os.path.exists('chi_square_top2000.xlsx'):
        # chi_square
        print('start chi_square method ...')
        start_time = time.time()
        keywords_chi_square = chi_square(DATA_PATH, DICT_PATH, corpus)
        print(keywords_chi_square)
        print('total time taken:', time.time() - start_time, 's')

    if not os.path.exists('word2vec_huffman_top2000_multiprocessing.xlsx'):
        # word2vec extra data processing
        categories = len(corpus)
        for i in range(categories):
            corpus[i] = corpus[i].split()

        # word2vec_huffman_softmax
        print('start huffman method ...')
        start_time = time.time()
        keywords_huffman = word2vec_huffman(corpus, MODEL_HUFFMAN_PATH)
        print(keywords_huffman)
        print('total time taken:', time.time() - start_time, 's')

    tf_idf_res = pd.read_excel('tf_idf_top2000.xlsx', sheet_name=None)
    word2vec_res = pd.read_excel('word2vec_huffman_top2000_multiprocessing.xlsx', sheet_name=None)
    chi_square_res = pd.read_excel('chi_square_top2000.xlsx', sheet_name=None)
    sheet_names = list(tf_idf_res.keys())
    sheet_num = len(sheet_names)
    writer = pd.ExcelWriter('intersection.xlsx')
    for i in range(sheet_num):
        new_sheet = []
        this_tf_idf = tf_idf_res[sheet_names[i]]
        this_word2vec = word2vec_res[sheet_names[i]]
        this_chi_square = chi_square_res[sheet_names[i]]
        for word in this_word2vec.values.tolist():
            word = word[0]
            if word not in new_sheet:
                new_sheet.append(word)
        for word in this_chi_square.values.tolist():
            word = word[0]
            if word not in new_sheet:
                new_sheet.append(word)
        for word in this_tf_idf.values.tolist():
            word = word[0]
            if word not in new_sheet:
                new_sheet.append(word)
        intersection = pd.DataFrame({'word': new_sheet})
        intersection.to_excel(writer, sheet_name=sheet_names[i], index=None)
        writer.save()
        writer.close()

    print('total time taken:', time.time() - start_time, 's')

    res_path = 'intersection.xlsx'

    return res_path


def main():
    global stopwords
    print('start processing data ...')
    start_time = time.time()
    stopwords = generateStopwords(STOPWORDS_PATH)
    corpus = data_processing(DATA_PATH, STOPWORDS_PATH, DICT_PATH)
    print('jieba tokenizing cost:', time.time() - start_time, 's')

    # tf-idf
    # print('start tf-idf method ...')
    # start_time = time.time()
    # keywords_tf_idf = tf_idf(corpus)
    # print(keywords_tf_idf)
    # print('total time taken:', time.time() - start_time, 's')

    # extra data processing for word2vec
    # categories = len(corpus)
    # for i in range(categories):
    #     corpus[i] = corpus[i].split()
    # print(corpus)

    # word2vec_kmeans
    # print('start kmeans method ...')
    # start_time = time.time()
    # keywords_kmeans = word2vec_kmeans(corpus, MODEL_KMEANS_PATH)
    # print(keywords_kmeans)
    # print('total time taken:', time.time() - start_time, 's')
    # print(corpus)

    # word2vec_huffman_softmax
    # print('start huffman method ...')
    # start_time = time.time()
    # keywords_huffman = word2vec_huffman(corpus, MODEL_HUFFMAN_PATH)
    # print(keywords_huffman)
    # print('total time taken:', time.time() - start_time, 's')

    # chi_square
    # print('start chi_square method ...')
    # start_time = time.time()
    # keywords_chi_square = chi_square(DATA_PATH, DICT_PATH, corpus)
    # print(keywords_chi_square)
    # print('total time taken:', time.time() - start_time, 's')


if __name__ == '__main__':
    # main()
    app.run(host='127.0.0.1', port=5000, debug=False, processes=True)


