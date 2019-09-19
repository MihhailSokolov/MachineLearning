"""
This is a k-Nearest Neighbours (k-NN) classifier for Fisher's Iris dataset
In this example k = 5 because it was shown in experimental way that k equals from 4 to 6 gives the best accuracy
Built during Lab 2 of Machine Learning course (CSE2510) in TUDelft by Mihhail Sokolov
"""
from sklearn import datasets # to load the dataset
from sklearn.model_selection import train_test_split #to split in train and test set
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import numpy as np
from collections import Counter #to count unique occurances of items in array, for majority voting

def euclidean(p, q):
    """
    Computes the euclidean distance between point p and q.
    :param p: point p as a numpy array.
    :param q: point q as a numpy array.
    :return: distance as float.
    """
    return distance.euclidean(p, q)

def get_neighbours(training_set, test_instance, k):
    """
    Calculate distances from test_instance to all training points.
    :param training_set: [n x d] numpy array of training samples (n: number of samples, d: number of dimensions).
    :param test_instance: [d x 1] numpy array of test instance features.
    :param k: number of neighbours to return.
    :return: list of length k with neighbour indices.
    """
    
    distances = np.zeros(len(training_set))
    
    for i, training_instance in enumerate(training_set):
        # Compute the distance to each item in the training set
        distances[i] = euclidean(test_instance, training_instance)
        
    # Return only k closest neighbours
    neighbours = np.argsort(distances)[:k]
    
    return neighbours


def get_majority_vote(neighbours, training_labels):
    """
    Given an array of nearest neighbours indices for a given test case, 
    tally up their classes to vote on the correct class for the test instance.
    :param neighbours: list of nearest neighbour indices.
    :param training_labels: the list of labels for each training instance.
    :return: the label of most common class.
    """
    labels = []
    for neighbour in neighbours:
        labels.append(training_labels[neighbour])
    counter = Counter(labels)
    max_vote = 0
    max_label = ''
    for label in counter.elements():
        if counter[label] > max_vote:
            max_vote = counter[label]
            max_label = label
    return max_label

def predict(X_train, X_test, y_train, y_test, k=5):
    """
    Predicts all labels for the test set, using k-nn on the training set and computes the accuracy.
    :param X_train: the training set features.
    :param X_test: the test set features.
    :param y_train: the training set labels.
    :param y_test: the test set labels.
    :return: list of predictions.
    """
    # generate predictions
    predictions = []
    # for each instance in the test set, get nearest neighbours and majority vote on predicted class
    for x in X_test:
        predictions.append(get_majority_vote(get_neighbours(X_test, x, k), y_test))
    return predictions

# load the data and create the training and test sets
iris = datasets.load_iris()
accuracies = []
iterations = 100
for i in range(iterations):
	X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4)
	k = 5
	predictions = predict(X_train, X_test, y_train, y_test, k)
	accuracy = accuracy_score(y_test, predictions)
	accuracies.append(accuracy)
# count average accuracy over all iterations
accuracy = np.mean(accuracies)
print('The overall accuracy of the model is: {:f}'.format(accuracy))