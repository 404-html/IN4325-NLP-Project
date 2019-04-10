# Main file that is used for performing experiments.
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

from confusion_matrix import plot_confusion_matrix
from data_processing import get_data
from metric_labeling import metric_labeling, train_nnc
from polarity_feature import make_polarity_features
from util import Author
from sampling import undersample, oversample, split_validation
from tune_params import tune_params

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

vectorizer_tf = TfidfVectorizer(max_features=12000)
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

X_features_train = np.hstack(X_features_train, get_features(full_sentences_train))
X_features_test = np.hstack(X_features_test, get_features(full_sentences_test))

y = data['class'].values
print("\nClass values: ")
print(str(y))

print("\nFeature array: ")
print(X_combined_train)

X_combined_train_under, labels_train_under = undersample(X_combined_train, labels_train)
X_combined_val, labels_val, X_combined_test_, labels_test_ = split_validation(X_combined_test, labels_test)

# SVC OVO
clf = svm.SVC(gamma=0.95, C=1.5, decision_function_shape='ovo')
clf.fit(X_combined_train, labels_train)
y_predicted = clf.predict(X_combined_test)
print(accuracy_score(labels_test, y_predicted))
plot_confusion_matrix(labels_test, y_predicted, np.array(('0', '1', '2')))
plt.show()

# Metric labeling
PSP_array_train = X_features_train[:,0].reshape(-1, 1).astype(float)
PSP_array_test = X_features_test[:,0].reshape(-1, 1).astype(float)

clf = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')
clf.fit(PSP_array_train, labels_train)
preferences = clf.decision_function(PSP_array_train)

plot_decision_regions(X=PSP_array_train,
                      y=np.array(labels_train),
                      clf=clf,
                      legend=2)
plt.show()

nnc = train_nnc(PSP_array_train, 5)
y_predicted = metric_labeling(PSP_array_train, labels_train, PSP_array_test, preferences, possible_labels, nnc)
print(accuracy_score(labels_test, y_predicted))
plot_confusion_matrix(labels_test, y_predicted, np.array(('0', '1', '2')))



X_features_under, labels_under = undersample(X_features_train, labels_train)
#Originally: 284 + 339 + 198 = 821

X_features_val, labels_features_val, X_features_test, labels_features_test = split_validation(X_features_test, labels_test)
best_params = tune_params(X_features_under, labels_under, X_features_val, labels_features_val, verbose=True)


clf_features = svm.SVC(??????????)
clf_features.fit(X_features_under, labels_under)
y_features_predicted = clf_features.predict(X_features_test)
print(accuracy_score(labels_features_test, y_features_predicted))
print(f1_score())
plot_confusion_matrix(labels_features_test, y_features_predicted, np.array(('0', '1', '2')))