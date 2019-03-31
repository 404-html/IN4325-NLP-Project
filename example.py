from sklearn import datasets

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

print("---------START Iris dataset SVM example---------")
X, y = datasets.load_iris(return_X_y=True)

# X containing feature values of data
print("Sepal length; Sepal Width; Petal Length; Petal Width")
print("Number of documents: " + str(len(X)))
print(X)
print("-------------")
# y containing classes of the previous data in order
print("Length of class data: " + str(len(y)))
print(y)

# C defines how much misclassification is allowed, low C = more misclassification and
# high C = less misclassification on test data. Be aware, high C will overfit
#
# Large Gamma would be too accurate and will overfit
m = svm.SVC(C=1.0, gamma='auto')

# If there are more than 2 classes in y, SVC will use one-vs-all by default
m.fit(X, y)
# Returns the mean accuracy on the given test data and labels. Using same train
# and test data right now (should be avoided).
print("Score: " + str(m.score(X, y)))
# Classes 0, 1 and 2
print("Different classes: " + str(m.classes_))