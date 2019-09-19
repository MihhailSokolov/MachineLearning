'''
This is a univariate Bayes Classifier for Iris flowers based on petal length
Built during Lab 2 of Machine Learning course (CSE2510) in TUDelft by Mihhail Sokolov
'''
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split #to split in train and test set

def compute_mean(x):
    return np.sum(x) / len(x)
    
def compute_sd(x, mean):
    return np.sqrt(np.sum(np.power(np.subtract(x, mean), 2)) / len(x))

# load the data and create the training and test sets
iris = datasets.load_iris()
# X is the feature vectors for the data points, and y is the target (ground truth) class for those data points 
#  the iris.data and iris.target entries are randomly divided into training and validation sets.
X_train, X_validation, y_train, y_validation = train_test_split(iris.data, iris.target, test_size=0.3) 

# Separate the dataset into the three flower types.
x_0 = X_train[np.where(y_train == 0)]
x_1 = X_train[np.where(y_train == 1)]
x_2 = X_train[np.where(y_train == 2)]

# Compute the mean for each flower type.
mean_0 = compute_mean(x_0[:, 2])
mean_1 = compute_mean(x_1[:, 2])
mean_2 = compute_mean(x_2[:, 2])

# Compute the standard deviation for each flower type.
sd_0 = compute_sd(x_0[:, 2], mean_0)
sd_1 = compute_sd(x_1[:, 2], mean_1)
sd_2 = compute_sd(x_2[:, 2], mean_2)

def normal_PDF(x, mean, sd):
    return 1 / np.sqrt(2 * np.pi * (sd ** 2)) * np.e ** (- np.power(np.subtract(x, mean), 2) / (2 * (sd ** 2)))

def posterior(x, means, sds, i):
    """
    Compute the posterior probabiliy P(C_i | x).
    :param x: the sample to compute the posterior probability for.
    :param means: an array of means for each class.
    :param sds: an array of standard deviation values for each class.
    :param i: the index of the class to compute the posterior probability for.
    """
    # First we compute the probability density function for class i
    pdf = normal_PDF(x, means[i], sds[i])
    
    # Next, we compute the sum of pdfs and use this to calculate the posterior probability
    all_pdfs = [normal_PDF(x, means[index], sds[index]) for index in range(3)]
    return pdf / np.sum(all_pdfs)

means = [mean_0, mean_1, mean_2]
sds = [sd_0, sd_1, sd_2]

def classify(x, means, sds):
    post_c1 = posterior(x, means, sds, 0)
    post_c2 = posterior(x, means, sds, 1)
    post_c3 = posterior(x, means, sds, 2)
    values = [post_c1, post_c2, post_c3]
    return values.index(max(values))

def validate(X_validation, target, means, sds):
    correct_predictions = 0
    total_predictions = len(X_validation)
    for i in range(total_predictions):
        if target[i] == classify(X_validation[i], means, sds):
            correct_predictions += 1
    return correct_predictions / total_predictions

print("Classifier accuracy is", validate(X_validation[:, 2], y_validation, means, sds))