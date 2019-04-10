# Main file that is used for performing experiments.
import re

import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn import model_selection

from confusion_matrix import plot_confusion_matrix
from data_processing import get_data, clean_sentences
from metric_labeling import metric_labeling, train_nnc
from polarity_feature import get_polarity_features
from rest_features import get_features
from sampling import undersample, oversample, split_validation
from tune_params import tune_params
from util import Author
from scipy.sparse import csr_matrix
import pandas as pd

# Load preprocessed data.
data, sentences = get_data(Author.SCHWARTZ)

y = data.iloc[:, 1].values
possible_labels = np.unique(y)
# Split training data
indices = range(len(sentences))
data_train, data_test, labels_train, labels_test, i_train, i_test = train_test_split(sentences, y, indices,
                                                                    test_size=0.20,
                                                                    random_state=42)
full_sentences_train = [re.findall('.*?[.!\?#]', data.iloc[i, 2] + "#") for i in i_train]
full_sentences_test = [re.findall('.*?[.!\?#]', data.iloc[i, 2] + "#") for i in i_test]

for review_sentences in full_sentences_train:
    if review_sentences[-1] == " #":
        review_sentences.pop()
    else:
        review_sentences[-1] = review_sentences[-1][:-1]
for review_sentences in full_sentences_test:
    if review_sentences[-1] == " #":
        review_sentences.pop()
    else:
        review_sentences[-1] = review_sentences[-1][:-1]

vectorizer_tf = TfidfVectorizer(max_features=8000)
# X_tf_train = vectorizer_tf.fit_transform(data_train)
# X_tf_test = vectorizer_tf.transform(data_test)

# This vocabulary can be extended with other words
### my_vocabulary = ["?"]
### vectorizer_voc = CountVectorizer(vocabulary=my_vocabulary,
###                                 token_pattern=r"(?u)\b\w\w+\b|\?") # Would get rid of 1-letter words
vectorizer_voc = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
# X_voc_train = vectorizer_voc.fit_transform(data_train)
# X_voc_test = vectorizer_voc.transform(data_test)

X_combined_train = FeatureUnion([('TfidfVectorizer', vectorizer_tf), ('CountVectorizer', vectorizer_voc)])
X_combined_train = X_combined_train.fit_transform(data_train).todense()

X_combined_test = FeatureUnion([('TfidfVectorizer', vectorizer_tf), ('CountVectorizer', vectorizer_voc)])
X_combined_test = X_combined_test.transform(data_test).todense()

X_features_train = get_polarity_features([clean_sentences(s) for s in full_sentences_train])
X_features_test = get_polarity_features([clean_sentences(s) for s in full_sentences_test])

X_features_train = np.hstack((X_features_train, get_features(full_sentences_train)))
X_features_test = np.hstack((X_features_test, get_features(full_sentences_test)))

y = data['class'].values
print("\nClass values: ")
print(str(y))

print("\nFeature array: ")
print(X_combined_train)

X_combined_train = csr_matrix(X_combined_train)
X_combined_test = csr_matrix(X_combined_test)

# hyperparameter selection rbf ovo
param_grid = {'gamma': [0.1, 0.18, 0.2, 0.25], 'C': np.linspace(3.8, 4.2, 5)}
model = svm.SVC()
clf_rbf_ovo = model_selection.GridSearchCV(model, param_grid, cv=10, return_train_score=True)
clf_rbf_ovo.fit(X_combined_train, labels_train)
print (clf_rbf_ovo.best_params_)
print (clf_rbf_ovo.best_score_)
search_results_rbf_ovo = pd.DataFrame.from_dict(clf_rbf_ovo.cv_results_)
print(search_results_rbf_ovo.loc[:, ['param_gamma', 'param_C', 'mean_test_score']])
#Best parameters {'C': 3.93, 'gamma': 0.18}

# hyperparameter selection linear ovo
param_grid = {'C': np.linspace(1, 5, 5)}
model = svm.SVC(kernel='linear')
clf_linear_ovo = model_selection.GridSearchCV(model, param_grid, cv=10, return_train_score=True)
clf_linear_ovo.fit(X_combined_train, labels_train)
print (clf_linear_ovo.best_params_)
print (clf_linear_ovo.best_score_)
search_results_linear_ovo = pd.DataFrame.from_dict(clf_linear_ovo.cv_results_)
print(search_results_linear_ovo.loc[:, ['param_C', 'mean_test_score']])
#Best parameters {'C': 3.93, 'gamma': 0.18}

# hyperparameter selection linear ova
param_grid = {'C': np.linspace(0.6, 3, 10)}
model = svm.LinearSVC()
clf_linear_ova = model_selection.GridSearchCV(model, param_grid, cv=10, return_train_score=True)
clf_linear_ova.fit(X_combined_train, labels_train)
print (clf_linear_ova.best_params_)
print (clf_linear_ova.best_score_)
search_results_linear_ova = pd.DataFrame.from_dict(clf_linear_ova.cv_results_)
print(search_results_linear_ova.loc[:, ['param_C', 'mean_test_score']])
#Best parameters {'C': 3.93, 'gamma': 0.18}


# SVC OVO
clf = svm.SVC(gamma=0.2, C=3.2, decision_function_shape='ovo')
clf.fit(X_combined_train, labels_train)
X_combined_train_under, labels_train_under = undersample(X_combined_train, labels_train)
X_combined_val, labels_val, X_combined_test_, labels_test_ = split_validation(X_combined_test, labels_test)

# SVC OVO
clf = svm.SVC(gamma='auto', C=1.5, decision_function_shape='ovo')
clf.fit(X_combined_train_under, labels_train_under)
y_predicted = clf.predict(X_combined_test)
print(accuracy_score(labels_test, y_predicted))
plot_confusion_matrix(labels_test, y_predicted, np.array(('0', '1', '2')), normalize=True)
plt.show()

# Metric labeling
PSP_array_train = X_features_train[:,0].reshape(-1, 1).astype(float)
PSP_array_test = X_features_test[:,0].reshape(-1, 1).astype(float)
PSP_array_train = [[x[0], 1-x[0]] for x in PSP_array_train]
PSP_array_test = [[x[0], 1-x[0]] for x in PSP_array_test]

clf = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')
clf.fit(X_combined_train, labels_train)
preferences = clf.decision_function(X_combined_train)

nnc = train_nnc(PSP_array_train, k=2)
y_predicted = metric_labeling(PSP_array_train, labels_train, PSP_array_test, preferences, possible_labels, nnc, alpha=100)
print(accuracy_score(labels_test, y_predicted))
plot_decision_regions(X=PSP_array_train,
                      y=np.array(labels_train),
                      clf=clf,
                      legend=2)
plt.show()

plot_confusion_matrix(labels_test, y_predicted, np.array(('0', '1', '2')))
plt.show()

#X_features_under, labels_under = undersample(X_features_train, labels_train)
X_features_over, labels_over = oversample(X_features_train, labels_train)
#Originally: 284 + 339 + 198 = 821
#Undersampling: 198 + 198 + 198 = 594
#Oversampling: 339 + 339 + 339 = 1017

X_features_val, labels_features_val, X_features_test_, labels_features_test = split_validation(X_features_test, labels_test)
best_params = tune_params(X_features_over, labels_over, X_features_val, labels_features_val, verbose=True)

plt.close('all')
print(best_params)

clf_features = svm.SVC(kernel='poly', degree=2, gamma=0.1, C=0.1, decision_function_shape='ovo')
clf_features.fit(X_features_over, labels_over)
y_features_predicted = clf_features.predict(X_features_test_)
print(accuracy_score(labels_features_test, y_features_predicted))
print(f1_score())
plot_confusion_matrix(labels_features_test, y_features_predicted, np.array(('0', '1', '2')))
