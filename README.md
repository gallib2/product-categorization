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

## Explanations of the included files
