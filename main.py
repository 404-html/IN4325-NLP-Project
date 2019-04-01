import csv
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer

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


create_csv_for(author_name, "|")

print("Read scaledata from data.csv file:\n")
df = pd.read_csv("data.csv", sep="|", header=0)
print("Dataframe: ")
print(df.head())

my_vocabulary = ["?"]
vectorizer = CountVectorizer(vocabulary=my_vocabulary,
                             token_pattern=r"(?u)\b\w\w+\b|\?")
X = vectorizer.transform(df['content'].tolist()).todense()
X_columns = vectorizer.get_feature_names()

# Generate PSP and last sentence sentiment features
print("\nStarting to make sentiment features, this will take some time...")
tb = Blobber(analyzer=NaiveBayesAnalyzer())

PSP_array = []
last_sentence_sentiment_array = []
for index, row in df.iterrows():
    blob = tb(row["content"])

    last_sentence_sentiment = 0
    if blob.sentences[-1].sentiment.p_pos > 0.5:
        last_sentence_sentiment = 1
    else:
        last_sentence_sentiment = 0

    positive_sentence_num = 0
    for index, sentence in enumerate(blob.sentences):
        if sentence.sentiment.p_pos > 0.5:
            positive_sentence_num += 1
    PSP = positive_sentence_num / len(blob.sentences)
    PSP_array.append([PSP])
    last_sentence_sentiment_array.append([last_sentence_sentiment])
print("Finished making sentiment features.")

# Append these features to the original feature matrix
X = np.hstack((X, np.asmatrix(PSP_array)))
X = np.hstack((X, np.asmatrix(last_sentence_sentiment_array)))

# Add feature names as well
X_columns = X_columns + ["PSP", "last_sentence_sentiment"]

print("\nFeature names: ")
print(X_columns)

y = df['class'].values
print("\nClass values: ")
print(str(y))

print("\nFeature array: ")
print(X)

m = svm.SVC(C=1.0, gamma='auto')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

m.fit(X_train, y_train)
# Classes 0, 1 and 2
print("\nDifferent classes: " + str(m.classes_))
print("Score: " + str(m.score(X_test, y_test)))
