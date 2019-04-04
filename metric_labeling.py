# Defines the metric labeling function.

from sklearn.metrics.pairwise import cosine_similarity;


# Default similarity function.
# Note that we can also use different ones, but the paper uses the cosine.
def sim(x, y):
    return cosine_similarity(x, y)


# Performs the metric labeling.
# TODO pass a custom similarity function.
def metric_labeling(training_set, test_set, preference, alpha=0.2, k=3):
    return True
