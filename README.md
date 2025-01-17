# Product categorization - Sears project

## Description
The objective of this project is to apply classification learning models on Sears products dataset with more than 100,000 
products, and therefore to obtain a predictive model with high accuracy for identifying future products categories.

### The dataset
The features in the datasets, among other things, include the name of the product, its description, price, image etc. 
Our dataset contains numeric values, string values and images.
### Goal
The project is aimed to categorize products according to existing metadata (name,
description, price, image) to a given taxonomy, by building ML algorithm that
classifies products according to their data.

Our main challenges were:
* How to convert the string values to a numeric vectors, and what is the better way to do so?
* How to deal with a large amount of data?
* Which classifier is the best to accomplish the project objective?

### The Solution
In order to classify the products that contain numeric values, string values and images,
we used two different classifiers - SVM and KNN.
The first step was to convert the string values to numeric vectors.
As a part of the converting process, we used Bag Of Words and several different sparse matrix - binary, tf, tf-idf, and sublinear tf-idf.

## Some results on the textual data
* #### SVM ([linear SVM from scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html))
	* Dataset weight: Tf
	* Data distribution: Stratified cross validation with k folds (k = 6)
	
	The results with different C parameters:
	![exp1](https://user-images.githubusercontent.com/22280734/32327116-06c94d38-bfde-11e7-8d51-19fe4bcbebe4.PNG)
	
	
	![Tf-write_answers__c-2^-5](https://user-images.githubusercontent.com/22280734/32329308-59365622-bfe5-11e7-86cd-85439f5fcce5.JPG)
	
	* Dataset weight: Tf-Idf (sublinear tf scaling, i.e. replace tf with 1 + log(tf))
	* Data distribution: Stratified cross validation with k folds (k = 6)
	
	The results with differnt C parameters:
	![exp2](https://user-images.githubusercontent.com/22280734/32327117-06e82262-bfde-11e7-9242-3a4e88953096.PNG)
	
	![Tf-Idf_sublinear_write_answers](https://user-images.githubusercontent.com/22280734/32329220-0bf3abee-bfe5-11e7-9ec4-5fac7740b7b1.JPG)
	
	* Dataset weight: Tf-Idf
	* Data distribution: Stratified cross validation with k folds (k = 6)
	
	The results with different C parameters:
	![exp3](https://user-images.githubusercontent.com/22280734/32327118-07073ee0-bfde-11e7-9cff-34024d72a9b5.PNG)
	
	![Tf-Idf_write_answers_c-2^-5](https://user-images.githubusercontent.com/22280734/32329162-d63b6afa-bfe4-11e7-82ae-0f202cf1761a.JPG)

	
	* Dataset weight: Binary
	* Data distribution: Stratified cross validation with k folds (k = 6)
	
	The results with different C parameters:
	![exp4](https://user-images.githubusercontent.com/22280734/32327115-06a8e764-bfde-11e7-8576-10f02ecfe450.PNG)
	
	![Binary-write_answers_c-2^-5](https://user-images.githubusercontent.com/22280734/32329299-5271f3fa-bfe5-11e7-994d-1afdbae67433.JPG)


* #### KNN 
	* Dataset weight: HashVectorizer
	* Data distribution: 20%-80% randomly
	* Search method: Nearest neighbors
	* Distance method: Cosine
	* Number of neighbors: 5
	
	The results:
	
	![exp1](https://user-images.githubusercontent.com/22280734/32327928-ef9ed558-bfe0-11e7-880e-655a61ea28cd.PNG)
	
	* Dataset weight: CountVectorizer
	* Data distribution: 20%-80% randomly
	* Search method: Nearest neighbors
	* Distance method: Cosine
	
	The results:
	
	![exp2](https://user-images.githubusercontent.com/22280734/32327929-efbfe6b2-bfe0-11e7-833d-b8091e072adc.PNG)
	
	* Dataset weight: Tf-Idf with ignoring numerical values
	* Data distribution: Stratified cross validation with k folds (k = 5)
	* Search method: K Neighbors Classifier
	* Distance method: Euclidean
	
	The results:
	
	![exp3](https://user-images.githubusercontent.com/22280734/32327931-efe0311a-bfe0-11e7-9ceb-7b25341adea4.PNG)
	
	* Dataset weight: Tf-Idf
	* Data distribution: Stratified cross validation with k folds (k = 5)
	* min_d: 1 (min_d – Create a dictionary composed of all the words that appear in the minimum min_d documents)
	
	The results:
	
	![exp4](https://user-images.githubusercontent.com/22280734/32327932-effe7116-bfe0-11e7-84e3-dee80dcd9b05.PNG)
	
	* Dataset weight: CountVectorizer
	* Data distribution: Stratified cross validation with k folds (k = 5)
	* min_d: 1 (min_d – Create a dictionary composed of all the words that appear in the minimum min_d documents)
	
	The results:
	
	![exp5](https://user-images.githubusercontent.com/22280734/32327933-f0234590-bfe0-11e7-90a9-f92735fc78f7.PNG)
	
	* Dataset weight: HashingVectorizer
	* Data distribution: Stratified cross validation with k folds (k = 5)
	* min_d: 1 (min_d – Create a dictionary composed of all the words that appear in the minimum min_d documents)
	
	The results:
	
	![exp6](https://user-images.githubusercontent.com/22280734/32327934-f04cdbda-bfe0-11e7-9c9b-d1aeec4431d1.PNG)
	
	
## Explanations of the included files

#### Note :
The files `'products_clean_144K_only_name_desc_clean.rar'`, `'All_products_clean.rar'`, `'products_all_cleanest.rar'` , which are in the main folder
and `'products_all_cleanest.rar'`, `'products_clean_144K.rar'`, `'products_clean_144K_price.rar'`, which are in the “csvfiles” folder.
These are CSV files, which must be extracted before using.

* ### clear_csv.py

    #### When to use
      Use this file first.

    #### How to use
      Open the terminal in the file’s location, and write:
      'python clear_csv.py -f products.csv'
      (when the 'products.csv' is the csv file to clean)

    #### The file contains:
Main function that clears the csv file from Special Characters.

	
* ### clean_csv.py
    #### When to use
      Use this file after "clear_csv.py".

    #### How to use
      Change the 'file_name_load' and 'file_name_save' variables to the wanted csv file name.

    #### The file contains:
      This file was created to make some additional cleanups of the csv file, in order to improve the SVM,
      and contains a function that:
       1. Removes HTML Tags
       2. Removes non-letters
       3. Converts to lower case, splits into individual words
       5. Removes stop words
       6. Stems words

* ### csvfixes.py

    #### When to use
      Use this file after "clear_csv.py" and "clean_csv.py".
      This file was created to make some additional cleanups of the csv file, in order to improve the SVM.

    #### How to use
      Change the 'file_name_load' and 'file_name_save' variables to the wanted csv file name.

    #### The file contains
      The file contains three main functions:
      1. "remove_irrelevant_docs" - Removes documents that belong to categories with frequency < 5 
      2. "match_number_to_category_id" – Adds a new column ("CategoryNumber") which sequentially matches a number to each category id (from 1 to category size).
         
      3. "remove_unique_words" - Removes words that appear only once in the entire document
        (this function is not relevant in case of using "max features" in Bag Of Words function)
      4. "remove_special_chars" - Removes given characters from the entire document.
  
* ### svm_linear.py
    #### When to use
      Use this file after using "clear_csv.py", "clean_csv.py" and "csvfixes.py"

    #### How to use
      In this file, change:
      1. The 'file_name_load' variable to the wanted csv file name.
      2. The 'c_parameter' variable to the wanted value, this is the cost parameter for the SVM.
      3. The 'type_matrix' variable of the suitable type of the sparse matrix that creates in the function 'set_cv_fit'
        This parameter is for the files name that are created after the classification

      NOTE: in the function 'set_cv_fit' - you can change the inner function 'CountVectorizer'
        to 'TfidfVectorizer' (just replace the comments).

    #### The file contains
      The linear SVM classification function
      The classification is made with stratified cross validation with k folds.
      When the classifier ends, it print the results to the following files:
      1. Predictions file - the predictions of the SVM, and the right label on the right (Comma separated)
      2. Statistics file that contains:
        - The number of correct answers of the classifier.
        - The number of incorrect answers of the classifier.
        - The percent of the right answers.
      3. Two files with a list of the categories numbers, and the number of correct/incorrect answers of each category (so that we can get some additional statistics on the results).
		


* ### dataframe_utils
      class of functions dealing with dataframe, mainly name, description and category fields.
      Use this class for actions and dataframe manipulations such as: load, remove, index, row/column extraction etc.

      "get_features_and_labels": returns concatenated features (name+description) and labels.
      "load_and_split_data": loads csv and splits to 80%-20% train-test
      "reset_index": after splitting data, reindex sequentially
      "remove_docs_by_column": remove documents of given column and column-value
      "remove_rare_categories_after_split" (unused): remove documents of unique targets after splittind tada.
      "rows_index_by_column": get dataframe index of documents which contains given values in a given column
      "find_rare_targets": return list of labels who appear less than 'max' times
	
* ### knn.py
    #### When to use
      Use for knn classification.

    #### How to use
      Class variables:

      filename: csv filename to create DataFrame from.

      type of vectorizer:
      TFIDF: for TfidfVectorizer
      COUNT_VEC: for CountVectorizer
      HASH_VEC: for HashingVectorizer

      vectorizer arguments:
      min_d: min_df parameter for vectorizing
      max_f: max_features (for CountVectorizer and TfidfVectorizer) / n_features (for HashingVectorizer)
      k_clusters: k_clusters for pysparnn.cluster_index.MultiClusterIndex classifier
      nbrs: n_neighbors for KNeighborsClassifier 
      metric: distance metric for KNeighborsClassifier

    see documentation:
    
    [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.predict)

   [feature_extraction](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer.transform)

     [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)

    [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

    https://github.com/facebookresearch/pysparnn

    #### The file contains
      Usage of Python's scikit-learn knn classifier: KNeighborsClassifier
      and PySparNN library for Nearest Neighbor approximation search aimed for sparse data.

      Class variables:


      vector creation:
      ----------------
      parameters:
      1. train
      2. test
      3. min_d
      4. max_f
      5. selection: TFIDF / COUNT_VEC / HASH_VEC (mentioned above)


      Classification:
      ---------------
      1. KNeighborsClassifier: train(vectorized), test(vectorized), train labels, neighbors, distance metric
      2. PySparNN: train(vectorized), test(vectorized), train labels, k_clusters

* ### vectorize.py
    #### When to use
      Class responsible for vectors and/or dataset creation, before classification.
      This class is calles from 'knn' class or 'svm'

      1. types of vectorization:
        1. TfidfVectorizer
          related functions:
            - create_vec_tfidf
            - fit_tfidf
            - create_vec_test
        2. CountVectorizer
          related functions:
            - create_vec_cv
            - fit_cv
            - create_vec_test
        3. HashingVectorizer
          related functions:
            - create_vec_hv
            - hash_feature

      2. svm format conversion:
        Args:
          features:  sparse matrix, shape [n_samples, n_features] 
          labels: labels corresponding to features.
          filename: file to write the vectors to.

        out of the above arguments, "write_vec", "write_libsvm_format", and "construct_line_libsvm_format"
        will format the dataset into svm acceptable shape, and write to file.

#### Resources
[API design for machine learning software: experiences from the scikit-learn project](https://arxiv.org/abs/1309.0238), Buitinck et al., 2013.

[Multi-Class Support Vector Machine](https://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html)
