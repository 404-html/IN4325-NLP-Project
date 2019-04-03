###POLARITY FEATURES
###################################
###################################

from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer

# Generate PSP and last sentence sentiment features
# Assumes data as an array of sentences
def make_polarity_features(data):
    print("\nStarting to make sentiment features, this will take some time...")
    PSP_array = []
    last_sentence_sentiment_array = []

    tb = Blobber(analyzer=NaiveBayesAnalyzer())
    for review in data:
        blob = tb(review)

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
    return PSP_array, last_sentence_sentiment_array