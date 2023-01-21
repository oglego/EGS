Author: Aaron Ogle

Project: Analysis of Machine Learning Algorithms for Classifying Electrical Grid Stability

Description: The attached source codes do the following,

AOKNN.py - 

This program is our implementation of the KNN algorithm; the program will read through the
txt versions of the data set and will then apply the KNN algorithm to the data in order to classify it.

Currently, the number of neighbors is set to 3, however this can be updated.  
The program contains comments throughout that should hopefully explain what each 
section of code is doing.

KNN.py -

This program is the scikit learn implementation of the KNN algorithm.  This program utilizes the csv file 
format of the data set and utilizes the scikit learn library in order to apply the KNN algorithm to the
data for classification.   

Along with classifying the data, the program also tests different values for K in order to find an optimal value.
The program utilizes scikit learn functions in order to display the models accuracy, precision, recall, and 
f1 score.  The program is very similiar to the AOKNN.py program, except that it has been modified so that it
can use the scikit learn functions.

AOLOGIT.py -

This program is our implementation of the Logistic Regression algorithm.  The program will read the
data set from its csv file, apply the Logistic Regression algorithm to the data, and then we will
test the accuracy of the model.  Currently, the program is set to run for 100 epochs, display the cost of
each epoch, until it finishes and will then display the accuracy of the model.  

LOGIT.py -

This program is the scikit learn implementation of the Logistic Regression algorithm.  This program utilizes
the scikit learn libraries accuracy, recall, precision, and f1 score functions so that these values can be
displayed.  The model is currently set to read the data from the csv, fit the Logistic Regression model
to the data, and then display the metrics.

ANN.py -

This program is our Keras implementation of a ANN algorithm.  The program reads the data set from its
csv file, cleans the data/standardizes the data, and then applies the ANN keras model to the data.  The program
is currently set to calculate the precision of the model, where the model is using the relu activation function
in the hidden layer, along with utilizing 100 neurons in the hidden layer.  Different sections of the code
can be commented in and out in order to perform different tests.

plots.py -

This program is a very simple helper program that was utilized to plot the results that were obtained when we
performed different tests on the programs discussed above.

Running the programs:

I am using the spyder python IDE in order to run the programs, however, as long as the below 
libraries have been installed then I believe the programs should run without having to use
the spyder IDE:

matplotlib.pyplot
keras.models
keras.layers
keras.optimizers
tensorflow
numpy
sklearn
sklearn.linear_model
sklearn.metrics
sklearn.neighbors
operator








