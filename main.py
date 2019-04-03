import csv
import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

author_name = "Dennis+Schwartz"


def create_csv_for(author, delimiter="|"):
    base_path = Path(__file__).parent
    target_dir = (base_path / "data/scaledata/" / author).resolve()
    column_names = ["id", "class", "content"]
    with io.open("data.csv", 'w', newline='') as fh_csv, \
            open(os.path.join(target_dir, "id." + author)) as fh1, \
            open(os.path.join(target_dir, "label.3class." + author)) as fh2, \
            open(os.path.join(target_dir, "subj." + author)) as fh3:

        writer = csv.writer(fh_csv, delimiter=delimiter)
        writer.writerow(column_names)

        while True:
            out = []
            for fh in [fh1, fh2, fh3]:
                out.append(fh.readline().strip('\n'))

            if all(out):
                writer.writerow(out)
            else:
                break


# Uncomment to create data.csv for author each time
# create_csv_for(author_name, "|")

# Create dataframe from data.csv
data = pd.read_csv("data.csv", sep="|", header=0)
# print(data.head())

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def clean_sentences(data):
    reviews = []

    for sent in data['content']:
        # remove html content
        review_text = BeautifulSoup(sent, "lxml").get_text()

        # remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z?]", " ", review_text)

        # tokenize the sentences
        words = word_tokenize(review_text.lower())

        # lemmatize each word to its lemma
        lemma_words = [lemmatizer.lemmatize(i) for i in words if
                       not i in set(stopwords.words('english'))]
        lemma_string = " ".join(lemma_words)
        reviews.append(lemma_string)

    return reviews


sentences = clean_sentences(data)
y = data.iloc[:, 1].values
# Split training data
data_train, data_test, labels_train, labels_test = train_test_split(sentences, y,
                                                                    test_size=0.20,
                                                                    random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_tf = TfidfVectorizer(max_features=12000)
# X_tf_train = vectorizer_tf.fit_transform(data_train)
# X_tf_test = vectorizer_tf.transform(data_test)

# This vocabulary can be extended with other words
my_vocabulary = ["?"]
vectorizer_voc = CountVectorizer(vocabulary=my_vocabulary,
                                 token_pattern=r"(?u)\b\w\w+\b|\?")
# X_voc_train = vectorizer_voc.fit_transform(data_train)
# X_voc_test = vectorizer_voc.transform(data_test)

from sklearn.pipeline import FeatureUnion

X_combined_train = FeatureUnion(
    [('TfidfVectorizer', vectorizer_tf), ('CountVectorizer', vectorizer_voc)])
X_combined_train = X_combined_train.fit_transform(data_train).todense()

X_combined_test = FeatureUnion(
    [('TfidfVectorizer', vectorizer_tf), ('CountVectorizer', vectorizer_voc)])
X_combined_test = X_combined_test.transform(data_test).todense()

# X_combined_train = X_tf_train
# X_combined_test = X_tf_test

from polarity_feature import make_polarity_features
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

from sklearn import svm

clf = svm.SVC(gamma=0.95, C=1.5, decision_function_shape='ovo')
clf.fit(X_combined_train, labels_train)

y_predicted = clf.predict(X_combined_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, y_predicted))

from confusion_matrix import plot_confusion_matrix

plot_confusion_matrix(labels_test, y_predicted, np.array(('0', '1', '2')))

