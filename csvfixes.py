import pandas as pd
import csv
from collections import Counter
import dataframe_utils as df_utils

filename_load = "csvfiles\\products_clean_144K.csv"
filename_save = "csvfiles\\products_clean_144K.csv"


def remove_special_chars(old_file, new_file, special_chars):
    with open(old_file, "r") as infile, open(new_file, "w") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, lineterminator='\n')
        conversion = set(special_chars)
        for row in reader:
            newrow = [''.join('' if c in conversion else c for c in entry) for entry in row]
            writer.writerow(newrow)


def remove_irrelevant_docs(df):
    """ Remove irrelevant documents """
    category = df["categoryNumber"]
    categories_to_del = df_utils.find_rare_targets("categoryNumber", 5)
    index_list = df_utils.rows_index_by_column(category, categories_to_del, df)
    df = df_utils.remove_rows_by_index(df, index_list)
    df.to_csv(filename_save, index=False)
    return df


def match_number_to_category_id(df):
    id = df["categoryId"]
    num_categories = id.unique().size
    id_to_number = {x: i for x,i in zip(id.unique(), range(1, num_categories + 1))}
    d = [id_to_number.get(k) for k in id.values]
    df['categoryNumber'] = pd.Series(d)
    df.to_csv(filename_save, index=False)


def remove_unique_words(df):
    word_counts = Counter(word for i, line in df.iterrows() for word in ' '.join(list([line['name'], line['description']])).split())
    # word_counts = list(reversed(word_counts.most_common()))
    unique = [word for (word, count) in word_counts.items() if count == 1]

    for u in unique:
        df['description'] = df.description.str.replace(' ' + u + ' ', ' ', 1)
        df['name'] = df.name.str.replace(' ' + u + ' ', ' ', 1)
    df.to_csv(filename_save, index=False)


def print_stats(df):
    print("dimensions: ", df.shape)
    print("columns: ", df.columns.values)


def main():
    dta = pd.read_csv(filename_load)
    print_stats(dta)
    remove_irrelevant_docs(dta)
    match_number_to_category_id(dta)
    remove_unique_words(dta)


if __name__ == "__main__":
    main()




