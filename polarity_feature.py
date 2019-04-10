###POLARITY FEATURES
###################################
###################################

import numpy as np
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer


# Calculate the "neutrality" of a text.
# Both a small difference between the number of positive and negative words and
# a large ratio of neutral words are directly correlated with a high neutrality.
def neutrality(pos, neg, neu, alpha):
    aux = float(pos+neg)
    if aux==0:
        return 1
    else:
        return alpha*(1-abs(pos-neg)/aux) + (1-alpha)*neu/(aux+neu)


# Generate PSP, last sentence sentiment, word and sentence neutrality and POS features
# Assumes data as an array of sentences
def get_polarity_features(data):
    print("\nStarting to make sentiment features, this will take some time...")

    alpha = 0.8
    polar_tags = ['JJ','JJR','JJS','RB','RBR','RBS']

    num_reviews = len(data)

    PSP_array = [None] * num_reviews
    last_sentence_sentiment_array = [None] * num_reviews
    first_sentence_sentiment_array = [None] * num_reviews
    neutrality_sentence_array = [None] * num_reviews
    neutrality_word_array = [None] * num_reviews
    tagged_words_array = [None] * num_reviews

    sia = SentimentIntensityAnalyzer()
    tb = Blobber(analyzer=NaiveBayesAnalyzer())
    for i, review in enumerate(data):
        blob = tb('. '.join(review))
        num_sentences = float(len(blob.sentences)) # To ensure floating point division


        first_sentence_sentiment = blob.sentences[0].sentiment.p_pos
        last_sentence_sentiment = blob.sentences[-1].sentiment.p_pos

        positive_sentence_num = 0
        negative_sentence_num = 0
        for sentence in blob.sentences:
            if sentence.sentiment.p_pos > 0.5:
                positive_sentence_num += 1
            else:
                negative_sentence_num += 1
        PSP = positive_sentence_num / num_sentences
        

        positive_words = 0
        negative_words = 0
        neutral_words = 0
        tagged = 0
        for word in ''.join(review).split(' '):
            if pos_tag([word])[0][1] in polar_tags:
                tagged += 1

            compound = sia.polarity_scores(word)['compound']
            if compound > 0:
                positive_words += 1
            elif compound < 0:
                negative_words += 1
            else:
                neutral_words += 1

        PSP_array[i] = PSP
        last_sentence_sentiment_array[i] = last_sentence_sentiment
        first_sentence_sentiment_array[i] = first_sentence_sentiment
        neutrality_sentence_array[i] = abs(positive_sentence_num-negative_sentence_num)/num_sentences
        neutrality_word_array[i] = neutrality(positive_words, negative_words, neutral_words, alpha)
        tagged_words_array[i] = tagged / float(positive_words+negative_words+neutral_words)
    
    print("Finished making sentiment features.")

    return np.vstack((PSP_array, last_sentence_sentiment_array, first_sentence_sentiment_array, neutrality_sentence_array, neutrality_word_array, tagged_words_array)).T
