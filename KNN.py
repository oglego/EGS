# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:19:13 2022

@author: Aaron Ogle


###############################################################################
DESCRIPTION

This program will use the KNN algorithm on the
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

In this program, we first import the data from a csv and clean it so that
it can be represented by a matrix in python.

We then normalize the data by using standardization.

After the data has been normalized, we then utilize the sklearn
library in order to apply the KNN algorithm to our data and we perform
and experiment to see which value for "k" provides the highest level of
accuracy when classifying our data.

###############################################################################
"""

# Utilize the regular expression library to split our 
# input file on the "," character and new lines
import re 

# Import the sklearn library in order to utilize the
# knn algorithm
from sklearn.neighbors import KNeighborsClassifier

# Import sklearn f1 score functionality
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

# Utilize numpys array functionality
import numpy as np 

# Utilize matplotlibs plotting functionality 
import matplotlib.pyplot as plt



"""
read_file function

This function will accept a file name as input and will then
read the contents of a file into a list.  The data that we are
working with consists of "," characters and line breaks so we
want to split the data based off of those characters.  Since
we have two values to split on we utilize a regular expression
function to help with that.  After this is done, the last item
in the list contains a empty '' value so we want to remove that
value from the list utilizing pop(); after all of this is done 
we return our data list.
"""

def read_file(file):
    
    # Read the data file, in this particular file we want to use 
    # regular expression and split the file by new line and ","
    with open(file) as f:
        data = f.read()
        data_set = re.split(r',|!|\n', data)
    # In the above, a ' ' value is returned at the end of the list
    # and we want to remove it
    data_set.pop()
    return data_set

"""
Convert_data function

The convert_data function will accept a list of data as input
and will then iterate through the contents of the list updating
the data type of each item to type float.
"""

def convert_data(data):
    # Iterate through the given data and update each 
    # value in the data to type floag
    for i in range(0, len(data)):
        data[i] = float(data[i])
    return data

"""
data_matrix function

The data_matrix function will accept a list type as input
and will then construct a matrix representation of the
provided data.  Since the data in the provided file
consists of rows of 13 values we want to create a matrix
that contains rows of data that are of length 13 so that
we include all of the relevant rows/columns that we need.
"""

def data_matrix(lyst):
    # Create an empty list and append rows of length
    # 13 to it; this will create a matrix representation
    # of our data then return the matrix
    data_set = []
    while lyst != []:
        # Use slicing to add 13 values to each row, then
        # use slicing to set lyst equal to the values 
        # after those 13 values
        data_set.append(lyst[:13])
        lyst = lyst[13:]
    return data_set

"""
min max normalization function

The min max normalization function will normalize our data
based off of the below equation:
    
    x' = x - min(x) / max(x) - min(x)
    
once the data has been normalized the function will return
the matrix 
"""

def min_max_normalization(matrix):
    # With this dataset, we know that there are values that
    # are higher and lower than our class labels so we can
    # just call the min and max functions on the entire
    # dataset
    
    # Find min value in data
    min_x = np.min(matrix)
    
    # Find max value in data
    max_x = np.max(matrix)
    
    # Now we will iterate through all of the rows/columns
    # excluding the final column which is our class label
    # and apply the min max normalization process on the
    # feature data
    for row in matrix:
        for i in range(len(row)-1):
            row[i] = ((row[i] - min_x) / (max_x - min_x) )
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
train_test_split function

This function will split the data into two sets, one set is
for training and the other set is for testing.  We will utilize
80% of the data for testing which will be 8000 rows of our data, 
the remaining 20% will be for testing.
"""

def train_test_split(dataset):
    # Create lists to house our training and testing data
    train = []
    test = []
    # Utilize 80% of the data for training - rows [0,8000]
    # and 20% for testing - rows [8000, 10000], return the lists
    for row in range(0,8000):
        train.append(dataset[row])
    for row in range(8000,len(dataset)):
        test.append(dataset[row])
    return train, test

"""
features_targets function

The features_targets function will create lists that contain
the feature values for the provided input and a list that
contains the target value.  We split our data so that we can
train/test our model using the Keras library.  The features
list will contain the features of our data while the targets
list will contain the target label (1,0) of our data label.
The function iterates through the data set that it is provided
and adds all of the features to a list and all of the target
values to a separate list; it then returns these lists.
"""

def features_targets(dataset):
    # Create lists for our feature data and target vectors
    features = []
    targets = []
    # Iterate through each row of the dataset and add the
    # feature data to the features list and the target
    # value to the targets list then return the lists
    for row in dataset:
        features.append(row[0:12])
        targets.append(row[12])
    return features, targets

def main():
    
    # Call our read_file function to read the data into a list
    data = read_file("EGC.csv")
        
    # Call our convert_data function to convert all of the
    # values in our data to type int
    c_data = convert_data(data)
    
    # Store our data into a matrix (list of lists)
    data_set = data_matrix(c_data)
    
    # Now we are going to convert our matrix into a 
    # numpy array so that we can utilize the np max
    # and np min functions for our min_max_normalization function
    data_set = np.array(data_set)
    
    # Now we normalize the data using standardization
    #norm_data = min_max_normalization(data_set)
    std_norm_data = standardization(data_set)
    
    # Call our train_test_split function so that we 
    # can split the data into training and testing sets
    train, test = train_test_split(std_norm_data)
    
    """
    Now that the data has been cleaned and split into
    training and testing datasets, we want to split
    the data further so that we have our feature vectors
    in one list and our output target labels in another
    """
    
    # Call the features_targets function on our training
    # and test data so that we have lists of our feature
    # data and lists of our target output vector data
    train_features, train_targets = features_targets(train)
    test_features, test_targets = features_targets(test)
    
    """
    Convert the training features, targets, and testing 
    features and targets into numpy arrays 
    """
    train_features = np.asarray(train_features)
    train_targets = np.asarray(train_targets)
    test_features = np.asarray(test_features)
    test_targets = np.asarray(test_targets)
    
    """
    In the below section, we utilize sklearns KNN classifier
    functionality in order to find the optimal value for
    K in our KNN algorithm that provides the highest
    accuracy for classifying our test data.
    """
    # We want to find an optimal value for k in our
    # knn algorithm so create an array for different
    # values for k
    k = [i for i in range(1,21)]
    
    # Create an array to keep track of the accuracy
    # for the different numbers of neighbors
    accuracy_array = []
    
    # Iterate through each of the different values for
    # our neighbors and fit our training data then
    # score our test data to see what the accuracy
    # is for different values of k
    for i in range(len(k)):
        # Python indexes at zero so we want to use i+1
        knn = KNeighborsClassifier(n_neighbors=i+1)
        knn.fit(train_features, train_targets)
        accuracy_array.append(knn.score(test_features, test_targets))
        
    # In the below section print the values of the accuracy
    # array and then create a plot that shows the accuracy
    # for each of the values of k
            
    plt.plot(k, accuracy_array)
    plt.legend()
    plt.title("KNN V2 Results")
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    
    # Set opt_k equal to the value of k that gave the highest accuracy
    opt_val = max(accuracy_array)
    for i in range(len(accuracy_array)):
        if accuracy_array[i] == opt_val:
            # use i + 1 since python indexes at zero
            opt_k = i + 1
    
    # Create our model based off of the "best" value for k
    model = KNeighborsClassifier(n_neighbors=opt_k)
    
    # Train our model
    model.fit(train_features, train_targets)
    
    # Display ideal k value
    print(f"Ideal k is: {opt_k}")
    
    # Score the accuracy of our model
    accuracy = model.score(test_features, test_targets)
    print(f"Accuracy: {accuracy}")

    # Find the predictions so that we can use it for metrics
    predictions = model.predict(test_features)
    
    # use sklearn functionality to calculate f1 score
    f1 = f1_score(test_targets, predictions)
    print(f"F1 Score: {f1}")
    
    # use sklearn functionality to calculate recall
    recall = recall_score(test_targets, predictions)
    print(f"Recall: {recall}")
    
    # use sklearn functionality to calculate precision
    precision = precision_score(test_targets, predictions)
    print(f"Precision: {precision}")

main()