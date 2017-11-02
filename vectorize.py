from __future__ import print_function
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from scipy import sparse
import dataframe_utils as df_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
import numpy as np
from nltk.tokenize import RegexpTokenizer

NAME = 0
DESCRIPTION = 1
SPARSE = 0
DENSE = 1


def create_vec_hv(train, test, max_f=(2 ** 20)):
    n, d = (0, 1) if type(train) == np.ndarray else ('name', 'description')
    vectorizer_n, features_vec_n = hash_feature(train, n, max_f)
    vectorizer_d, features_vec_d = hash_feature(train, d, max_f)
    train_vec_ = hstack((features_vec_n, features_vec_d), dtype=np.float16)
    if type(train) == np.ndarray:
        test_vec_ = hstack((vectorizer_n.transform(test[:, n]), vectorizer_d.transform(test[:, d])), dtype=np.float16)
    else:
        test_vec_ = hstack((vectorizer_n.transform(test[n]), vectorizer_d.transform(test[d])), dtype=np.float16)
    return train_vec_, test_vec_


def hash_feature(data, feature, max_f):
    vectorizer = HashingVectorizer(n_features=max_f, stop_words='english', alternate_sign=False, norm='l1', dtype=np.float32)
    # testing make_pipeline
    # vectorizer = make_pipeline(hasher, TfidfTransformer())

    if type(data) == np.ndarray:
        vectorizer.fit(data[:, feature])
        features_vec = vectorizer.transform(data[:, feature])
    else:
        vectorizer.fit(data[feature])
        features_vec = vectorizer.transform(data[feature])
    return vectorizer, features_vec


def create_vec_tfidf(train, test, min_d=2, max_f=None):
    names, descriptions = df_utils.get_name_desc_columns(train)
    features_vec_n = fit_tfidf(names, NAME, min_d, max_f)
    features_vec_d = fit_tfidf(descriptions, DESCRIPTION, min_d, max_f)
    train_vec = hstack((features_vec_n, features_vec_d), dtype=np.float16)
    test_vec = create_vec_test(test, tv_n, tv_d)
    return train_vec, test_vec


def fit_tfidf(df_, column, min_d, max_f):
    global tv_n, tv_d

    tokenizer_ = RegexpTokenizer(r'\w+')
    tv = TfidfVectorizer(analyzer="word", min_df=min_d, max_features=max_f, sublinear_tf=False, norm='l1', tokenizer=tokenizer_.tokenize)
    tv.fit(df_)
    features_vec_ = tv.transform(df_)

    if column == NAME:
        tv_n = tv
    else:
        tv_d = tv

    return features_vec_


def create_vec_cv(train, test, min_d, max_f):
    names, descriptions = df_utils.get_name_desc_columns(train)
    features_vec_n = fit_cv(names, NAME, min_d, max_f)
    features_vec_d = fit_cv(descriptions, DESCRIPTION, min_d, max_f)
    train_vec = sparse.csr_matrix(hstack((features_vec_n, features_vec_d)))
    test_vec = create_vec_test(test, cv_name, cv_desc)
    return train_vec, test_vec


def fit_cv(df_, column, min_d, max_f):
    global cv_name, cv_desc

    cv = CountVectorizer(analyzer="word", min_df=min_d, max_features=max_f, binary=False)
    cv.fit(df_)
    features_vec_ = cv.transform(df_)

    if column == NAME:
        cv_name = cv
    else:
        cv_desc = cv

    return features_vec_


def create_vec_test(test, tv_n, tv_d):
    n_test, d_test = df_utils.get_name_desc_columns(test)
    search_features_vec = hstack((tv_n.transform(n_test), tv_d.transform(d_test)), dtype=np.float16)
    return search_features_vec


def write_vec(features, labels, filename):
    # list of each vector size
    sizes = [features.getrow(i).indptr[1] for i in range(features.shape[0])]
    write_libsvm_format(labels, sizes, features, filename)


def write_libsvm_format(id_, sizes, sparse_mat, file_to_write):
    o = open(file_to_write, 'w')
    sum = 0
    for i, size in enumerate(sizes):
        indices_i = sparse_mat.indices[sum:sum + size]
        data_i = sparse_mat.data[sum:sum + size]
        new_line = construct_line_libsvm_format(id_[i], indices_i, data_i)
        o.write(new_line)
        sum += size


def construct_line_libsvm_format(label, indices, data):
    new_line = [str(label)]
    for i, d in zip(indices, data):
        new_item = "%s:%s" % (i+1, d)
        new_line.append(new_item)
    new_line = " ".join(new_line)
    new_line += "\n"
    return new_line





    '''if df has index column 'Unnamed: 0' (usually for using Pandas.savetxt), do the following: '''
    # df = df.drop('Unnamed: 0', axis=1)

    ''' "category" column location'''
    # df.columns.get_loc("category")

    ''' list of columns names '''
    # list(df.columns.values)

    ''' df as matrix'''
    # np_df = df.as_matrix()
    # np_df[0]

    ''' save sparse matrix (as zipped file)'''
    # np.savez("sparse", data=sparse_mat.data, indices=sparse_mat.indices, indptr =sparse_mat.indptr, shape=sparse_mat.shape)


