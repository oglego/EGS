# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:19:13 2022

@author: Aaron Ogle

###############################################################################
DESCRIPTION

This program will use my own implementation of the logistic regression algorithm on the 
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

After the data has been normalized, we apply the logistic regression 
classification algorithm on the data in order to classify it.

###############################################################################
"""

# Utilize the regular expression library to split our 
# input file on the "," character and new lines
import re 

# Utilize numpys array functionality
import numpy as np 

import math # needed for sigmoid function

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

"""
*************************************************************************

activate function

The activate function when using logistic regression is the sigmoid
function so we define a similar activate function like we did with
the perceptron learning algorithm but we utilize the sigmoid function.

If the result of passing our summation into the sigmoid function is
greater than or equal to 0.5 then we label the class as 1, otherwise 
we label it as 0.

*************************************************************************
"""

def activate(s):
    
    z = (1.0 / (1.0 + math.exp(-s)))
    if z >= 0.5:
        z = 1.0
    else:
        z = 0.0
    return z

"""
*************************************************************************

feed_forward function

The feed forward function will accept rows of our training data matrix that 
we defined below in the main section along with a weights value.  The function
will sum the values of the given inputs multipled by the weights of the
inputs and will then return the value of passing the summation into our
activation function defined above.  Note that this function is based off of
the sample code provided in our lecture on the implementation of the 
perceptron learning algorithm.  We utilize the same function here for our
logistic regression model.

*************************************************************************
"""

def feed_forward(inputs, weights):
    # Set a sum value equal to zero so that we can perform a summation
    summ = 0.0
    # For every row in our matrix input we want to
    # sum the values of the inputs multiplied by the weights value
    # then we will call the activation function defined above and 
    # return that value.  Note here that we exclude the last column
    # in the inputs row as this value is the label and we do
    # not want to use that in our summation
    for i in range(len(inputs)-1):
        summ += inputs[i] * weights[i]
    return activate(summ)

"""
The logit function will iterate through a given number of epochs and will
apply the gradient descent algorithm in order to update our weights
based off of the sigmoid function results defined above in the activate
function.  After we have gone through all of the epochs the weights are
returned.
"""
def logit(training_data, epochs):
    # Initialize all weights to zero
    weights = [0.0 for i in range((len(training_data[0])))]

    # Set our learning rate to be 0.1
    n = 0.1
    
    for epoch in range(epochs):
        # The cost function for linear regression is defined as
        # 1/2*m * sum(predicted - true)**2
        # so we set a value equal to zero so that we can sum and
        # calculate this value to display after each epoch
        cost = 0
        # Iterate through each row of our matrix and call
        # the feed_forward function for the row along with
        # the values for the weights
        for row in training_data:
            # Call the feed forward function
            output = feed_forward(row, weights)
            # Our desired output is the label that is in the 
            # 12 column of our rows
            desired = row[12]
            # Calculate the cost
            cost += (desired - output)**2
            cost *= (1 /(2 * (len(row)-1))) 
            # Iterate through each value of the row excluding
            # the value in the last column as this is our
            # label for the row data
            for i in range(len(row)-1):
                # Utilize gradient descent in order to update the weights
                # where the gradient descent process is
                # weight[j] = weight[j] + (learning_rate)*sum(predicted-true)*x[j]
                # We set s equal to zero so we can sum the difference between
                # the true value and predicted value multiplied by the value of
                # x in our row, then we update our weights based off of 
                # multiplying our learning rate times the resulting sum and
                # the value of the current weight
                s = 0
                s += ((desired - output)*row[i])
                weights[i] += (n * s)   
                
                
        # Display the epoch that we are on and the cost
        print(f"Epoch: {epoch}  Cost: {cost}")
    # Return the values found for the weights
    return weights


"""
The logistic regression function will take our training and test data
along with number of epochs as input and will then find the value
of the weights based off of our training data.  Once we have the
weights we then apply our test data to those weights in order to
predict values.  We keep a list of these predicted values so that we
can calculate accuracy later on; the function returns this list.
"""
def logistic_regression(train, test, epoch):
    predict = []
    weights = logit(train, epoch)
    for row in test:
        pred = feed_forward(row, weights)
        predict.append(pred)
    return predict

"""
The accuracy function will calculate how accurate our logistic
regression model is.  We set a counter equal to zero and then
increment it each time that the model makes a correct
prediction.  We then return the total number of results that
the model got correct and divide this value by the total number.
"""
def accuracy(true, pred):
    counter = 0
    for i in range(len(true)):
        if true[i] == pred[i]:
            counter += 1
    return counter / float(len(true)) * 100.0
        

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

    # Normalize the data
    norm_data = standardization(data_set)
    #norm_data = min_max_normalization(data_set)
    # Call our train_test_split function so that we 
    # can split the data into training and testing sets
    train, test = train_test_split(norm_data)
    
    # Parse out all of the labels in our test data and add them
    # to a list so that we can test our model
    test_val = []
    for row in test:
        test_val.append(row[12])

    # Apply our training and test data to our model with 100 epochs
    log_reg = logistic_regression(train, test, 100)
    
    # Display the accuracy
    print(accuracy(test_val, log_reg))
    

main()