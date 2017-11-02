# Product categorization - Sears project

## Description
The objective of this project is to apply classification learning models on sears products dataset with more than 100,000 
products and therefore to obtain a predictive model with high accuracy for identifying future products categories.

### The dataset
The features in the datasets among other thing includes name of the product, his description, price, image, and etc. 

### The problem
This project is aimed to build a predictive model that is able to distinguish products between more then 1500 product categories.
A few selected classification learning models will be trained by the dataset that includes each product’s corresponding category.
Our dataset contains numeric values, string values and images.
Our main problems was:
* How to convert the string values to a numeric vectors, and what is the better way to do so
* How to deal with a large amount of data
* Which classifier is the best for accomplish project objective

### The Solution
In order to classify the products that contains numeric values, string values and images,
we use two different of classifiers - SVM and KNN,
although the first step was to convert the string values to numeric vectors.
Part of the converting processes includes using Bag Of Words and several different sparse matrix - binary, tf, tf-idf, and sublinear tf-idf.

## Some results on the textual data
* #### SVM ([linear SVM from scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html))
	* dataset weight: Tf
	* Stratified cross validation with k folds (k = 6)
	the results with differnt C parameters:
	![exp1](https://user-images.githubusercontent.com/22280734/32327116-06c94d38-bfde-11e7-8d51-19fe4bcbebe4.PNG)
	
	* dataset weight: Tf-Idf (sublinear tf scaling, i.e. replace tf with 1 + log(tf))
	* Stratified cross validation with k folds (k = 6)
	the results with differnt C parameters:
	![exp2](https://user-images.githubusercontent.com/22280734/32327117-06e82262-bfde-11e7-9242-3a4e88953096.PNG)
	
	* dataset weight: Tf-Idf
	* Stratified cross validation with k folds (k = 6)
	the results with differnt C parameters:
	![exp3](https://user-images.githubusercontent.com/22280734/32327118-07073ee0-bfde-11e7-9cff-34024d72a9b5.PNG)
	
	* dataset weight: Binary
	* Stratified cross validation with k folds (k = 6)
	the results with differnt C parameters:
	![exp4](https://user-images.githubusercontent.com/22280734/32327115-06a8e764-bfde-11e7-8576-10f02ecfe450.PNG)

## Explanations of the included files

#### Note :
The files `'products_clean_144K_only_name_desc_clean.rar'`, `'All_products_clean.rar'`, `'products_all_cleanest.rar'` that in the main folder
and `'products_all_cleanest.rar'`, `'products_clean_144K.rar'`, `'products_clean_144K_price.rar'`, that in the csvfiles folder
are the CSV files, and must to be extracted before using the files.

* ### clear_csv.py

    #### When to use
      use this file first.

    #### How to use
      open terminal in this file location, and write:
      'python clear_csv.py -f products.csv'
      (when the 'products.csv' is the csv file to clean)

    #### The file contains
	    main function that clears the csv file from Special Characters.

	
* ### clean_csv.py
    #### When to use
      use this file after "clear_csv.py".

    #### How to use
      in this file, change the 'file_name_load' and 'file_name_save' variables to the wanted csv file name.

    #### The file contains
      this file was created to make some additional cleanup of the csv file for improving the svm,
      and contains function that does:
       1. Remove HTML Tags
       2. Remove non-letters
       3. Convert to lower case, split into individual words
       5. Remove stop words
       6. stem words

* ### csvfixes.py

    #### When to use
      use this file after "clear_csv.py" and "clean_csv.py"
      this file was created to make some additional cleanup of the csv file for improving the svm

    #### How to use
      in this file, change the 'file_name_load' and 'file_name_save' variables to the wanted csv file name.

    #### The file contains
      this file contains 3 main functions:
      1. "remove_irrelevant_docs" - removes documents that belong to categories with frequency < 5 
      2. "match_number_to_category_id" - sequentially matches a number to category id (from 1 to category size),
         to an added new column: "CategoryNumber"
      3. "remove_unique_words" - removes words that appear only once in the entire document
        (this function is not relevant in case of using "max features" in Bag Of Words function)
      4. "remove_special_chars" - removes given characters from entire document.
  
* ### svm_linear.py
    #### When to use
      use this file after "clear_csv.py", "clean_csv.py" and "csvfixes.py"

    #### How to use
      in this file, change:
      1. 'file_name_load' variable to the wanted csv file name.
      2. 'c_parameter' variable to the wanted value, this is the cost parameter for the SVM.
      3. 'type_matrix' variable of the suitable type of the sparse matrix that creates in the function 'set_cv_fit'
        this paramter is for the the files name that creats after the classifictaion

      NOTE: in the function 'set_cv_fit' - you can change the inner function 'CountVectorizer'
        to 'TfidfVectorizer' (just replace the comments).

    #### The file contains
      this file contains the linear svm classifcation function
      the classification is made with stratified cross validation with k folds.
      when the classifier ends, it print the reault to the following files:
      1. predictions file - the predictions of the svm, and the right label on the right (Comma separated)
      2. statistics file that contains:
        - numer of correct answers of the classifier.
        - number of incorrect answers of the classifier.
        - percent of the right answers.
      3. two file with list of the categories number, and the number of correct/incorrect answers of each category
        (so we can get some additional statistics on the results)
		


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
