import scipy.sparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from scipy import sparse
import vectorize


def devide_to_chunks(test_v):
    chunks = []
    as_csr = test_v.tocsr()
    rows = test_v.shape[0]
    for i in range(0, rows, 500):
        if i < rows - 500:
            chunks.append(as_csr[i:i + 500, :])
        else:
            chunks.append(as_csr[i:, :])
    return chunks


def get_predictions_count(labels_len):
    file_name = "predictions_" + str(c_parameter) + type_matrix + ".csv"
    predictions = pd.read_csv(file_name)
    count_correct = 0
    count_incorrect = 0
    correct_arr = [0 for i in range(labels_len)]
    incorrect_arr = [0 for i in range(labels_len)]

    for i in range(predictions.shape[0]):
        predict_value = predictions.values[i][0]
        real_class_value = predictions.values[i][1]

        if predictions.values[i][0] != "end of fold":
            if int(predict_value) == int(real_class_value):
                count_correct = count_correct + 1
                correct_arr[int(real_class_value) - 1] += 1
            else:
                count_incorrect = count_incorrect + 1
                incorrect_arr[int(real_class_value) - 1] += 1

    percent_correct = (count_correct / (count_correct + count_incorrect)) * 100

    correct_str = "count_correct: " + str(count_correct)
    incorrect_str = "count_incorrect: " + str(count_incorrect)
    percent_str = "percent of correct:" + str(percent_correct)
    print(correct_str)
    print(incorrect_str)
    print(percent_str)

    file_name_result = "statistics_" + str(c_parameter) + type_matrix + ".txt"
    with open(file_name_result, "a") as file:
        file.write("%s\n%s\n%s" % (correct_str, incorrect_str, percent_str))

    file_name_correct_arr = "corrects_" + str(c_parameter) + type_matrix + ".txt"
    with open(file_name_correct_arr, "a") as file:
        for i in range(len(correct_arr)):
            file.write("%s,%s\n" % (i + 1, correct_arr[i]))

    file_name_incorrect_arr = "incorrects_" + str(c_parameter) + type_matrix + ".txt"
    with open(file_name_incorrect_arr, "a") as file:
        for i in range(len(incorrect_arr)):
            file.write("%s,%s\n" % (i + 1, incorrect_arr[i]))


def write_predict_to_file(pred, label_test):
    file_name = "predictions_" + str(c_parameter) + type_matrix + ".csv"
    with open(file_name, "a") as file:
        for i in range(len(pred)):
            file.write("%s,%s\n" % (pred[i], int(label_test[i])))
        file.write("end of fold\n")


def svm_stratified_cross_validation_kfolds(_samples, _labels):
    skf = StratifiedKFold(n_splits=6, shuffle=True)

    for train_index, test_index in skf.split(_samples, _labels):
        print("TRAIN:", train_index, "TEST:", test_index)
        sample_train, sample_test = _samples[train_index], _samples[test_index]
        label_train, label_test = _labels[train_index], _labels[test_index]

        sparse_train, sparse_test = vectorize.create_vec_cv(sample_train, sample_test, 2, 20000)

        clf = LinearSVC(random_state=0, C=c_parameter, dual=False)
        clf.fit(sparse_train, label_train)
        chunks = devide_to_chunks(sparse_test)
        predictions = [clf.predict(x) for x in chunks]
        predictions_flat = [item for sublist in predictions for item in sublist]
        write_predict_to_file(predictions_flat, label_test)

        # clf = LinearSVC(random_state=0, C=c_parameter, dual=False)
        # clf.fit(sparse_train, label_train)
        # write_predict_to_file(clf.predict(sparse_test), label_test)


# samples = np.concatenate((data["name"][:, None],data["description"][:, None]), axis=1)


# sparse_matrix = scipy.sparse.load_npz("sparse_matrix_sample_144K.npz")
# samples = sparse.csr_matrix(sparse_matrix)
# data = pd.read_csv("products_clean_144K_only_name_desc_clean.csv")
# labels = data["categoryNumber"].as_matrix()

c_parameter = pow(2, 0)
type_matrix = "_regular_"
file_name_load = "csvfiles\\products_clean_144K.csv"

data = pd.read_csv(file_name_load)
samples = np.concatenate((data["name"][:, None], data["description"][:, None]), axis=1)
labels = data["categoryNumber"].as_matrix()
cv_name, cv_desc = None, None

svm_stratified_cross_validation_kfolds(samples, labels)
get_predictions_count(labels.max())





# def get_columns(data):
#     names = []
#     descriptions = []
#     for i in range(len(data)):
#         names.append(data[i][0])
#         descriptions.append(data[i][1])
#
#     return names, descriptions

#
# def get_columns_by_number(data, index):
#     column_data = []
#     for i in range(len(data)):
#         column_data.append(int(data[i][index]))
#
#     return column_data


# def set_cv_fit(names, descriptions, is_train):
#     global cv_name, cv_desc
#
#     if is_train:
#         cv_name = CountVectorizer(analyzer="word", min_df=2, binary=False, max_features=20000)
#         #cv_name = TfidfVectorizer(analyzer="word", min_df=2,  max_features=20000)
#         cv_fit_n = cv_name.fit_transform(names)
#         cv_desc = CountVectorizer(analyzer="word", min_df=2, binary=False, max_features=20000)
#         #cv_desc = TfidfVectorizer(analyzer="word", min_df=2,  max_features=20000)
#         cv_fit_d = cv_desc.fit_transform(descriptions)
#     else:
#         # fit test according to train BOW
#         cv_fit_n = cv_name.transform(names)
#         cv_fit_d = cv_desc.transform(descriptions)
#
#     return cv_fit_n, cv_fit_d


# def create_dataset(df_, is_train):
#     # reset for sequential index values after splitting
#     names, descriptions = get_columns(df_)
#     #prices = get_columns_by_number(df_, 2)
#     fit_names, fit_desc = set_cv_fit(names, descriptions, is_train)
#     # concat matrices (name vectors + description vectors + ...)
#     #sparse_mat = sparse.csr_matrix(hstack((fit_names, fit_desc, np.asarray(prices)[:, None].astype(np.int16))), dtype=np.int16)
#     sparse_mat = sparse.csr_matrix(hstack((fit_names, fit_desc)), dtype=np.int16)
#     # list of each vector size
#     return sparse_mat
