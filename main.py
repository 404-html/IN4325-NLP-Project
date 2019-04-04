# Main file that is used for performing experiments.

import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

from confusion_matrix import plot_confusion_matrix
from data_processing import get_data
from polarity_feature import make_polarity_features
from util import Author

# Load preprocessed data.
data, sentences = get_data(Author.SCHWARTZ)

y = data.iloc[:, 1].values
# Split training data
data_train, data_test, labels_train, labels_test = train_test_split(sentences, y,
                                                                    test_size=0.20,
                                                                    random_state=42)

vectorizer_tf = TfidfVectorizer(max_features=12000)
# X_tf_train = vectorizer_tf.fit_transform(data_train)
# X_tf_test = vectorizer_tf.transform(data_test)

# This vocabulary can be extended with other words
my_vocabulary = ["?"]
vectorizer_voc = CountVectorizer(vocabulary=my_vocabulary,
                                 token_pattern=r"(?u)\b\w\w+\b|\?")
# X_voc_train = vectorizer_voc.fit_transform(data_train)
# X_voc_test = vectorizer_voc.transform(data_test)

X_combined_train = FeatureUnion(
    [('TfidfVectorizer', vectorizer_tf), ('CountVectorizer', vectorizer_voc)])
X_combined_train = X_combined_train.fit_transform(data_train).todense()

X_combined_test = FeatureUnion(
    [('TfidfVectorizer', vectorizer_tf), ('CountVectorizer', vectorizer_voc)])
X_combined_test = X_combined_test.transform(data_test).todense()

# X_combined_train = X_tf_train
# X_combined_test = X_tf_test


PSP_array_train, last_sentence_sentiment_array_train = make_polarity_features(data_train)
PSP_array_test, last_sentence_sentiment_array_test = make_polarity_features(data_test)

# Append these features to the original feature matrix
X_combined_train = np.hstack((X_combined_train, np.asmatrix(PSP_array_train)))
X_combined_train = np.hstack(
    (X_combined_train, np.asmatrix(last_sentence_sentiment_array_train)))

X_combined_test = np.hstack((X_combined_test, np.asmatrix(PSP_array_test)))
X_combined_test = np.hstack(
    (X_combined_test, np.asmatrix(last_sentence_sentiment_array_test)))

y = data['class'].values
print("\nClass values: ")
print(str(y))

print("\nFeature array: ")
print(X_combined_train)

clf = svm.SVC(gamma=0.95, C=1.5, decision_function_shape='ovo')
clf.fit(X_combined_train, labels_train)

y_predicted = clf.predict(X_combined_test)

print(accuracy_score(labels_test, y_predicted))

plot_confusion_matrix(labels_test, y_predicted, np.array(('0', '1', '2')))
