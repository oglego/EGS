#!/usr/bin/python



# -*- coding: utf-8 -*-

"""
Created on Fri Mar 11 09:19:13 2022

@author: Aaron Ogle


###############################################################################
DESCRIPTION

This program will use my implementation of the KNN algorithm on the
"Electrical Grid Stability Simulated Data Set" found on UCI's machine
learning repository.

We have cleaned the data set so that it contains the following features and
class labels:
    
    1) tau1 
    2) tau2 
    3) tau3 
    4) tau4  
    5) p1 
    6) p2 
    7) p3
    8) p4
    9) g1
    10) g2
    11) g3
    12) g4
    13) Class label - (stable or unstable)
    
The class label has been converted to a float value where 1.0 indicates "stable"
and 0.0 indicates "unstable".

From UCI's website, we can see that 

tau[x] is the reaction time of participants where tau1 is the value for 
electricity producer

p[x] is the power that is consumed (negative) and produced (positive) 

g[x] is the coefficient that is proportional to price elasticity; where g1
is the value for the electricity producer

The final column is the class label which has been discussed above.

In this program, we first import the data from a text file and clean it so that
it can be represented by a matrix in python.

After the data has been standardized, we apply the knn algorithm to the data
and compute how accurate the algorithm is based off of the data in the test
and training set.

###############################################################################
"""


import math
import random
# Import itemgetter in order to sort a matrix based off of value
# that is in the last column in the matrix
from operator import itemgetter

# Needed to get mean and std for standardization
import numpy as np

# Define a global constant for the number of features 
NUM_FEATURES = 12
# Define a global constant for the position of the class label
LABEL = 13

"""
The function  "construct_matrix" will accept as input rows of vectors;
these rows of vectors are from the input files provided to us.
It will then create a list of lists where the input is split on newlines.
This function will allow us to create a list of lists data structure so that
computations will be easier to implement.
"""

def construct_matrix(vectors):
    # Given that the input is a matrix, we want to create a list of lists
    # so that we can manipulate the data more easily
    # Go through each row in the provided input and split on the newline
    # then use *map to unpack the arguments of mapping type int to the
    # splitting of the different rows in the input
    # After this, return the list of lists which will be our matrix data structure
    matrix = [[*map(float, row.split())] for row in vectors.split('\n')]
    return matrix

"""
standardization function

The standardization function will standardize our data by
applying the below equation:
    
    x' = x - mean(x) / standard deviation(x)
"""

def standardization(matrix):
    # This function will standardize data by setting each value 
    # equal to x - mean(x) / std(x)
    #mean = np.average(matrix)
    #std = np.std(matrix)
    mean = []
    std = []
    # Use zip* to transpose
    for col in zip(*matrix):
        mean.append(np.average(col))
        std.append(np.std(col))
    for row in matrix:
        for i in range(len(row)-1):
            row[i] = ((row[i] - mean[i]) / std[i])
    return matrix

"""
The function "euclidean_distance" will accept two vectors as input
and will compute the euclidean distance between the two vectors.

Here again, since the class labels are in the test and training
dataset matrix we are going to ignore the last column of the vectors.
"""

def euclidean_distance(v1, v2):
    sum_of_squares = 0
    for i in range(len(v1)-1):
        sum_of_squares += (v1[i] - v2[i])**2
    euclid_d = math.sqrt(sum_of_squares)
    return euclid_d

"""
The function "knn" will accept a matrix, vector, and a value for "k"
which represents the number of "k" nearest neighbors the user wants
to use for their classification.  

The input matrix will be the training data, and the input vector will
be the test vector that the user wishes to classify.
"""

def knn(matrix, vector, k):
    # Create a list to keep track of the euclidean distances
    vector_distances = []
    num_rows = len(matrix)
    # Calculate the euclidean distance between vectors in the
    # input matrix and the vector that we want to classify
    for row in range(num_rows):
        vector_distances.append(euclidean_distance(matrix[row], vector))
    # Append the distances to a copy matrix so that we can utilize them
    copy_matrix = matrix
    for row in range(num_rows):
        copy_matrix[row].append(vector_distances[row])
    distance_location = len(copy_matrix[0])
    
    # Sort the vectors in the matrix by the distance between
    # them and the vector we want to classify
    sorted_matrix_by_dist = sorted(copy_matrix, key=itemgetter(distance_location-1))
    # Create a list for the number of neighbors we want to use
    # for classification then only input those "k" values into the list
    # return the list
    k_neighbors_list = []
    for i in range(k):
        k_neighbors_list.append(sorted_matrix_by_dist[i])
    return k_neighbors_list

def classify(matrix, vector_class):
    # The location of the class in the input matrix is in column 12
    CLASS = 12
    # Create a list so we can keep track of the predicted classes
    predicted_classes = []
    num_rows = len(matrix)
    for i in range(num_rows):
        predicted_classes.append(matrix[i][CLASS])
    # Compute the number of times the class appears in the above list
    class_frequency = {}
    for label in predicted_classes:
        class_frequency[label] = predicted_classes.count(label)
    # Now we need to iterate through the dictionary to find which class
    # or classes have the highest number of occurences
    highest_frequencies = 0
    class_list = []
    for key, value in class_frequency.items():
        if value > highest_frequencies:
            highest_frequencies = value
    # If the number of occurences is equal to the max number of
    # class occurences calculate above then add it to our class_list
    for key, value in class_frequency.items():
        if value == highest_frequencies:
            class_list.append(key)
    # If the length of the class_list is greater than one, meaning
    # we had a tie between different classes in the times that they
    # appeared, then randomly select one of them, otherwise
    # just return the class that was predicted the most
    if len(class_list) > 1:
        prediction = class_list[random.randint(0,len(class_list)-1)]
    else:
        prediction = class_list[0]
    # Now that we have something that can keep track of ties, we now
    # need to compute the accuracy, the below logic is based off of
    # the rules that were supplied to us on the definition of 
    # accuracy in the assignment
    if len(class_list) == 1 and vector_class == prediction:
        accuracy = 1
    elif len(class_list) == 1 and vector_class != prediction:
        accuracy = 0
    elif len(class_list) > 1 and prediction in class_list:
        accuracy = 1 / len(class_list)
    elif len(class_list) > 1 and prediction not in class_list:
        accuracy = 0
    return prediction, round(accuracy, 4) 


def main():
    
    # Uncomment in order to invoke the program by the command line
    """
    # Read in what the user has input into the command line
    training_data_file = sys.argv[1]
    test_data_file = sys.argv[2]
    num_of_neighbors = sys.argv[3]
    num_of_neighbors = int(num_of_neighbors)
    """
    """
    Read and store the data from the input files into variables.
    """
    
    # Uncomment to implement without command line
    training_data_file = "training.txt"
    test_data_file = "test.txt"
    
    
    # Read through the files
    with open(training_data_file) as f1:
        training_data = f1.read()
    
    with open(test_data_file) as f2:
        test_data = f2.read()
        
    
    
    # Create matrices from the input files
    training_matrix = construct_matrix(training_data)
    test_matrix = construct_matrix(test_data)
    
    # While constructing our matrices, there is an extra []
    # added in, and we want to remove these
    training_matrix.pop()
    test_matrix.pop()
    
    # Standardize the input matrices
    transform_train_m = standardization(training_matrix)
    transform_test_m = standardization(test_matrix)
    
    # Set num rows to the number of rows in the test data matrix
    num_rows = len(transform_test_m)
    
    # Set the variable m to be the training matrix after being normalized
    m = transform_train_m
    
    
    
    """
    In the below section, we will go through each of the vectors in the
    test dataset and compute the k nearest neighbors of that vector with
    all of the vectors that are in the training dataset.
    """
    
    # Set accuracy summ to zero so we can calculate the accuracy 
    summ = 0
    # Set number of neighbors
    num_of_neighbors = 3
    # Iterate through each of the vectors in the test data
    for i in range(num_rows):
        vect = transform_test_m[i]
        # Compute knn of the current vector with all vectors in the training data
        knn_result = (knn(m, vect, num_of_neighbors))
        # Set the result tuple equal to the classification tuple
        result_tup = classify(knn_result, vect[12])
        # Set prediction
        predicted = result_tup[0]
        # Set accuracy
        accuracy = result_tup[1]
        # Sum accuracies
        summ += accuracy
        print(f"\nID = {i}, Predicted = {predicted}, True = {vect[12]}")
        m = transform_test_m
    # Calculate overall accuracy
    accuracy = summ / num_rows
    # Move decimal place to get percentage
    accuracy = accuracy * 100
    
    print(f"Classification Accuracy = {accuracy}%")

"""

1 neighbor
Classification Accuracy = 91.45%

3 neighbors
Classification Accuracy = 74.3%

9 neighbors
Classification Accuracy = 67.45%


"""

main()


