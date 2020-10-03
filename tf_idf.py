import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


def tf_idf(corpus):
    categories = len(corpus)
    """
        CountVectorizer:
        a common feature value computation class
        Convert a collection of text documents to a matrix of token counts
        This implementation produces a sparse representation of the counts using scipy.sparse.csr_matrix.
    """
    vectorizer = CountVectorizer()
    # transform corpus into matrix
    word_frequency = vectorizer.fit_transform(corpus)
    keywords = vectorizer.get_feature_names()
    """
    Transform a count matrix to a normalized tf or tf-idf representation
    """
    transformer = TfidfTransformer(smooth_idf=False)
    # get tf_idf values
    # tf_idf[i][j]: tf_idf of jth word in ith document
    tf_idf = transformer.fit_transform(word_frequency).toarray()
    # print('shape:', tf_idf.shape)
    top2000_index = [[]]
    for i in range(categories - 1):
        top2000_index.append([])

    # top2000_words = set()
    top2000_words = []
    writer = pd.ExcelWriter('tf_idf_top2000.xlsx')
    for i in range(categories):
        top2000_index[i] = np.argsort(tf_idf[i])[::-1][:2000]
        this_top2000_values = np.zeros(2000).round(decimals=8)
        # this_top2000_words = set()
        this_top2000_words = []
        for j in range(2000):
            # this_top2000_words.add(keywords[top2000_index[i][j]])
            this_top2000_values[i] = tf_idf[i][j]
            # print('tf_idf:', tf_idf[i][j])
            # print('this_value', this_top2000_values[i])
            # print('--------------------------')
            this_top2000_words.append(keywords[top2000_index[i][j]])
        # print(tf_idf[i])
        for word in this_top2000_words:
            if word not in top2000_words:
                top2000_words.append(word)
        # print(this_top2000_words)
        top_word_value = pd.DataFrame({'word': this_top2000_words, 'tf_idf_value': this_top2000_values})
        top_word_value['tf_idf_value'] = top_word_value['tf_idf_value'].astype(float).round(decimals=8)
        # print(sorted(this_top2000_values, reverse=True))
        sheet_name = 'cate_' + str(i + 1)
        top_word_value.to_excel(writer, sheet_name=sheet_name, index=None, float_format='%.5f')

    top_words = pd.DataFrame({'word': top2000_words})
    top_words.to_excel(writer, sheet_name='4 cate', index=None)
    writer.save()
    writer.close()
    return top2000_words