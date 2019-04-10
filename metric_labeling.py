# Defines the metric labeling function.

from scipy import spatial
from sklearn.neighbors import NearestNeighbors


# Metric used for labels.
# Note that we can also use different ones, but the paper assumes this metric.
def d(x, y):
    return abs(x - y)


# Trivial f definition, which is a monotonically increasing function, based on the paper.
def f(x):
    return x


# Default similarity function.
# Note that we can also use different ones, but the paper uses the cosine.
def sim(x, y):
    return 1 - spatial.distance.cosine(x, y)


# Trains a nnc classifier.
def train_nnc(training_data, k=5):
    nnc = NearestNeighbors(k, metric='cosine')
    nnc.fit(training_data)
    return nnc


# Performs the metric labeling.
def metric_labeling(training_set, labels_train, test_set, preferences, possible_labels, nnc, alpha=0.2, k=5):
    labels = []
    for i, item in enumerate(test_set):
        costs = []
        for l in possible_labels:
            neighbor_cost_values = []
            neighbors = nnc.kneighbors([item], k, return_distance=False)
            neighbors = neighbors.tolist()[0]
            for n in neighbors:
                neighbor_item = training_set[n]
                neighbor_label = labels_train[n]
                neighbor_cost_values.append(f(d(l, neighbor_label)) * sim(item, neighbor_item))
            correct = sum(neighbor_cost_values)
            costs.append(-preferences[i][l] + alpha * correct)
        labels.append(possible_labels[costs.index(min(costs))])
    return labels


# Performs the metric labeling.
def metric_labeling_opt(args, training_set, labels_train, test_set, preferences, possible_labels):
    alpha, k = args
    print(args)
    print(k)
    cost = 0
    for i, item in enumerate(test_set):
        costs = []
        for l in possible_labels:
            neighbor_cost_values = []
            nnc = train_nnc(training_set, k)
            neighbors = nnc.kneighbors([item], k, return_distance=False)
            neighbors = neighbors.tolist()[0]
            for n in neighbors:
                neighbor_item = training_set[n]
                neighbor_label = labels_train[n]
                neighbor_cost_values.append(f(d(l, neighbor_label)) * sim(item, neighbor_item))
            correct = sum(neighbor_cost_values)
            costs.append(-preferences[i][l] + alpha * correct)
        costs += costs.index(min(costs))
    return cost
