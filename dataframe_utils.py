import numpy as np
import pandas as pd


def get_features_and_labels(data):
    features = np.concatenate((data["name"][:, None], data["description"][:, None]), axis=1)
    labels = data["categoryNumber"].as_matrix()
    return features, labels


def load_and_split_data(filename):
    df = pd.read_csv(filename, dtype={"categoryId": str, "categoryNumber": str})
    # df = np.genfromtxt(filename, dtype=None, delimiter=',', names=True, usecols=['name', 'description', 'categoryNumber'])
    train, test = split_df(df)
    # train, test = remove_rare_categories_after_split(train, test, "categoryNumber", max=2)
    return train, test


def split_df(df_):
    msk = np.random.rand(len(df_)) < 0.9
    return reset_index(df_[msk], df_[~msk])


def get_name_desc_columns(df_):
    if type(df_) == np.ndarray:
        return df_[:, 0], df_[:, 1]
    n = df_["name"]
    d = df_["description"]
    return n, d


def reset_index(train, test):
    """reset for sequential index values after splitting"""
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test


def remove_docs_by_column(data, column_name, column_values):
    index_train = rows_index_by_column(data[column_name], column_values, data)
    return remove_rows_by_index(data, index_train)


def remove_rare_categories_after_split(train, test, column_name, max):
    rare_categories = find_rare_targets(train, column_name, max)
    train = remove_docs_by_column(train, column_name, rare_categories)
    test = remove_docs_by_column(test, column_name, rare_categories)
    return train, test


def rows_index_by_column(column, column_values, data):
    """index-list of documents (rows) by category"""
    l = []
    for value in column_values:
        l.append(data.index[column == value].tolist())

    # flatten list of lists to a single list
    flat_l = sorted([item for sublist in l for item in sublist])
    # check if list has duplicates (if it does, something is wrong)
    # duplicates = len(flat_l) != len(set(flat_l))
    return flat_l


def remove_rows_by_index(data, index_list):
    """remove index-list's corresponding documents"""
    return data.drop(data.index[index_list])


def find_rare_targets(data, target_name, max):
    targets = pd.Categorical(data[target_name])

    # ordered by appearances (number of documents) (ascending)
    ordered = targets.value_counts().sort_values()

    # group by appearances
    grouped = ordered.groupby(ordered.values)

    ''' check statistics '''
    # for each category, check in how many documents appears
    # grouped.count()  # (x, y) ==> label x appears in y documents
    # g = dict(list(grouped))

    # group all categories with appearance < max
    group_to_del = pd.concat([grouped.get_group(group) for i, group in enumerate(grouped.groups) if i < max])

    # number of documents to remove
    group_to_del.values.sum()

    # list of categories to remove (appear in less then max docs/rows)
    rare_tergets = group_to_del.keys().tolist()
    return rare_tergets
