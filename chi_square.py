import pandas as pd
import jieba
import math


punc = "，。、【 】:“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥*"
STOPWORDS_PATH = 'data/stop_words_ch.txt'


def generateStopwords(path):
    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    return stopwords


# judge if chinese
def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def word_in_cate(corpus, cate, word):
    # return A+C, A
    text_num_in_cate = len(corpus[cate])
    count = 0
    for i in range(text_num_in_cate):
        if word in corpus[cate][i]:
            count += 1
    return text_num_in_cate, count


def word_in_all(corpus, word):
    # return A+B
    count = 0
    for cate in corpus:
        text_num_in_cate = len(corpus[cate])
        for i in range(text_num_in_cate):
            if word in corpus[cate][i]:
                count += 1
    return count


def word_all_not_in_cate(corpus, cate):
    # return B+D
    # del corpus[cate]
    count = 0
    for other_cate in corpus:
        if other_cate != cate:
            count += len(corpus[other_cate])
    return count

stopwords = generateStopwords(STOPWORDS_PATH)


def chi_square(data_path, dict_path, corpus):

    data_raw = pd.read_excel(data_path, header=None)
    data = {}
    categories = []
    num = len(data_raw)
    jieba.load_userdict(dict_path)
    for i in range(num):
        category = int(data_raw[0][i])
        pre_data = data.get(category, [])
        if category not in categories:
            categories.append(category)

        this_data = []
        content = jieba.lcut(data_raw[1][i])
        for word in content:
            if word not in stopwords and word not in punc and is_chinese(word) and len(word) > 1:
                this_data.append(word)
        pre_data.append(this_data)
        data[category] = pre_data

    categories = len(corpus)
    if type(corpus[0]) != str:
        for i in range(categories):
            corpus[i] = " ".join(corpus[i])
    for i in range(categories):
        corpus[i] = corpus[i].split()
        # print(type(corpus[i]))

    top_words = []
    writer = pd.ExcelWriter('chi_square_top2000.xlsx')
    for i in range(categories):
        word_chi = []
        cate = i + 1
        corpus_cate = corpus[i]
        for word in corpus_cate:
            chi = get_chi(data, cate, word)
            word_chi.append((word, chi))
        word_chi = sorted(list(word_chi), key=lambda x: x[1], reverse=True)
        this_top_word = [word for word, chi in word_chi[:2000]]
        this_top_value = [chi for word, chi in word_chi[:2000]]
        top_word_value = pd.DataFrame({'word': this_top_word, 'value': this_top_value})
        print(i+1, top_word_value)
        sheet_name = 'cate_' + str(i + 1)
        top_word_value.to_excel(writer, sheet_name=sheet_name, index=None)
        for word in this_top_word:
            if word not in top_words:
                top_words.append(word)

    final_top_words = pd.DataFrame({'word': top_words})
    final_top_words.to_excel(writer, sheet_name=str(categories) + ' cate', index=None)
    writer.save()
    writer.close()
    return top_words


def get_chi(data, cate, word):
    # A+C
    all_word_in_this_cate = word_in_cate(data, cate, word)[0]
    # A
    this_word_in_this_cate = word_in_cate(data, cate, word)[1]
    # A+B
    this_word_in_all_cate = word_in_all(data, word)
    # B+D
    all_word_in_other_cate = word_all_not_in_cate(data, cate)
    # B
    this_word_in_other_cate = this_word_in_all_cate - this_word_in_this_cate
    # C
    other_word_in_this_cate = all_word_in_this_cate - this_word_in_this_cate
    # D
    other_word_in_other_cate = all_word_in_other_cate - this_word_in_other_cate
    rough_chi = math.pow(this_word_in_this_cate * other_word_in_other_cate - this_word_in_other_cate * other_word_in_other_cate, 2) / (
        (this_word_in_all_cate + 0.01) * (other_word_in_this_cate + other_word_in_other_cate + 0.01))
    # print(rough_chi)
    return rough_chi