import re
import inline as inline
import matplotlib as matplotlib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords  # Import the stop word list
from nltk.stem.lancaster import LancasterStemmer
from seaborn import set_style
from sklearn.feature_extraction.text import CountVectorizer

matplotlib, inline
pd.set_option("max_rows", 10)
np.set_printoptions(suppress=True)
# desired_width = 320
# pd.set_option('display.width', desired_width)

set_style("darkgrid")
file_name_load = "All_products_clean.csv"
file_name_save = "products_all_cleanest.csv"

dta = pd.read_csv(file_name_load)
print("dta.head():\n", dta.head())
print("dta.info():\n", dta.info())
print("dta.describe():\n", dta.describe())


def product_str_to_words(raw_desc):
    # The input is a single string (a raw product name/description), and
    # the output is a single string (a preprocessed product name/description)
    #
    # 1. Remove HTML
    desc_text = BeautifulSoup(raw_desc).get_text()

    # 2. Remove non-letters
    letters_numbers_only = re.sub("[^a-zA-Z\d]", " ", desc_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_numbers_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if w not in stops]

    st = LancasterStemmer()
    meaningful_words = [st.stem(w) for w in meaningful_words]

    # 6. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)


# Get the number of data length based on the dataframe column size
num_desc = dta["description"].size

# Initialize an empty list to hold the clean data
clean_dta_desc = []

# Loop over each name/description; create an index i that goes from 0 to the length
# of the csv file
for i in range(0, num_desc):
    # Call our function for each one, and add the result to the list of
    # clean names and descriptions
    try:
        dta["description"][i] = product_str_to_words(dta["description"][i])
        dta["name"][i] = product_str_to_words(dta["name"][i])
        print(dta["name"][i])
        print(i)
        #clean_dta_desc.append(review_to_words(dta["name"][i]))
    except TypeError:
        pass


dta.to_csv(file_name_save, encoding='utf-8')


