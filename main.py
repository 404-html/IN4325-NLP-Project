import csv
import io
import os
from pathlib import Path

import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

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
vectorizer = CountVectorizer(binary=True, vocabulary=my_vocabulary,
                             token_pattern=r"(?u)\b\w\w+\b|\?")
X = vectorizer.transform(df['content'].tolist())
y = df['class'].values
print(str(y))

print("Feature array: ")
print(X.toarray())

m = svm.SVC(C=1.0, gamma='auto')

m.fit(X, y)
print("Score: " + str(m.score(X, y)))
# Classes 0, 1 and 2
print("Different classes: " + str(m.classes_))
