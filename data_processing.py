# Defines functions used for data processing related tasks.

import csv
import io
import os
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

data_dir = "./processed_data/"


def clean_sentences(data):
    lemmatizer = WordNetLemmatizer()
    reviews = []

    # for sent in data['content']:
    for sent in data:
        # remove html content
        review_text = BeautifulSoup(sent, "lxml").get_text()

        # remove non-alphabetic characters
        review_text = re.sub("[^a-z ]", "", review_text)

        # tokenize the sentences
        words = word_tokenize(review_text) # Reviewes are already in lower-case

        # lemmatize each word to its lemma
        lemma_words = [lemmatizer.lemmatize(i) for i in words if i not in set(stopwords.words('english'))]
        lemma_string = " ".join(lemma_words)
        reviews.append(lemma_string)

    return reviews


# Creates a csv file, given the relevant author and an optional delimiter.
def create_csv_for(author, delimiter="|"):
    author = author.value
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


def get_data(author):
    if not os.path.isfile(data_dir + "data.csv"):
        create_csv_for(author)
    data = pd.read_csv(data_dir + "data.csv", sep="|", header=0)
    # sentences = clean_sentences(data)
    sentences = clean_sentences(data['content'])
    return data, sentences
