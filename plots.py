# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:16:45 2022

@author: Aaron

Plots - this is a simple helper program to just plot some of the
results that were obtained
"""
import matplotlib.pyplot as plt

AOKNN_accuracy = [91.45, 74.3, 67.45]
k = [1, 3, 9]
    
plt.plot(k, AOKNN_accuracy)
plt.title("KNN V1 Results")
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


sig_acc = [79.6, 85.6, 94.55, 92.05]
relu_acc = [79.7, 84.5, 95.4, 95.7]
neurons = [1, 5, 50, 100]

plt.plot(neurons, sig_acc, 'red', label="Sigmoid")
plt.plot(neurons, relu_acc, 'green', label="ReLU")
plt.xlabel("Neurons")
plt.ylabel("Accuracy")
plt.title("Sigmoid vs Relu Accuracy Results")
plt.legend(loc='upper left')
plt.show()