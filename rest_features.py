import numpy as np
def get_features(data):
    num_reviews = len(data)

    
    num_words_array = [None] * num_reviews
    wps_array = [None] * num_reviews # Average words per sentence
    num_question_array = [None] * num_reviews
    num_exclamation_array = [None] * num_reviews
    num_quote_array = [None] * num_reviews
    num_parenthesis_array = [None] * num_reviews

    for i, review in enumerate(data):

        num_words = 0
        question = 0
        quote = 0
        exclamation = 0
        parenthesis = 0

        for sentence in review:
            words = sentence.split(' ')
            for word in words:
                if len(word)>0:
                    if word[0].isalpha():
                        num_words += 1
                    elif word[0]=='?':
                        question += 1
                    elif word[0]=='"':
                        quote += 1
                    elif word[0]=='!':
                        exclamation += 1
                    elif word[0]=='(' or word[0]==')':
                        parenthesis += 1

        sentences = float(len(review))
        num_words_array[i] = np.log(num_words)
        wps_array[i] = np.log(num_words / sentences)
        num_question_array[i] = question / sentences
        num_exclamation_array[i] = exclamation / sentences
        num_quote_array[i] = quote / (2*sentences)
        num_parenthesis_array[i] = parenthesis / (2*sentences)

    return np.vstack((num_words_array, wps_array, num_question_array, num_exclamation_array, num_quote_array, num_parenthesis_array)).T