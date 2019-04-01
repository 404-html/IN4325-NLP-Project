import csv
import io
import os
from pathlib import Path
from string import punctuation

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def create_polarity_dataset(delimiter="|"):
    base_path = Path(__file__).parent
    target_dir = (base_path / "data/polaritydata").resolve()
    column_names = ["class", "content"]
    with io.open("polarity_data.csv", 'w', newline='') as fh_csv:
        writer = csv.writer(fh_csv, delimiter=delimiter)
        writer.writerow(column_names)
        for filename in os.listdir(target_dir / "neg"):
            with open(os.path.join(target_dir / "neg", filename)) as fh1:
                for line in fh1:
                    out = [0, line.strip('\n')]
                    writer.writerow(out)
        for filename in os.listdir(target_dir / "pos"):
            with open(os.path.join(target_dir / "pos", filename)) as fh1:
                for line in fh1:
                    out = [1, line.strip('\n')]
                    writer.writerow(out)


create_polarity_dataset(delimiter="|")


# ------------------------------------------------------------------------

class PreProcessPolarityDataset:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation))

    def processPolaritySentences(self, df_of_sentences):
        processedSentences = []
        print("Starting to tokenize polarity data...")
        for index, row in df_of_sentences.iterrows():
            processedSentences.append(
                (self.processPolaritySentence(row["content"]), row["class"]))
        print("Tokenizing finished!")
        return processedSentences

    def processPolaritySentence(self, sentence):
        tokenized_sentence = word_tokenize(sentence)
        return [word for word in tokenized_sentence if word not in self._stopwords]


polarityProcessor = PreProcessPolarityDataset()

df_of_sentences = pd.read_csv("polarity_data.csv", sep="|", header=0)
preprocessedSentences = polarityProcessor.processPolaritySentences(df_of_sentences)


# ------------------------------------------------------------------------
def buildVocabulary(preprocessedSentences):
    all_words = []

    for (words, sentiment) in preprocessedSentences:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()

    return word_features


# ------------------------------------------------------------------------

def extract_features(sentence):
    sentence_words = set(sentence)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in sentence_words)
    return features


# ------------------------------------------------------------------------

# Now we can extract the features and train the classifier
word_features = buildVocabulary(preprocessedSentences)
trainingFeatures = nltk.classify.apply_features(extract_features, preprocessedSentences)

# ------------------------------------------------------------------------

print("Starting to train...")
NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
print("Trainting finished!")

# ------------------------------------------------------------------------

NBResultLabels = []
for index, row in df_of_sentences.iterrows():
    NBResultLabels.append(NBayesClassifier.classify(extract_features(row["content"])))

print(NBResultLabels)