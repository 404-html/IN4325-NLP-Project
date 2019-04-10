import numpy as np

def undersample(X, y):
    if not type(X) is np.array:
        X = np.array(X)
    classes = len(np.unique(y))
    # Assumes the class-labeles are integers from 0 to C-1, the total number of classes
    original_classes = [np.sum(y==i) for i in range(classes)]
    minority = np.min(original_classes)

    X_under = []
    y_under = []
    under_classes = [0 for _ in range(classes)]

    for i, x in enumerate(X):
        class_i = y[i]
        if under_classes[class_i] < minority:
            X_under.append(x)
            y_under.append(class_i)
            under_classes[class_i] += 1
    return (np.matrix(X_under), np.array(y_under))

def oversample(X, y):
    return True

def split_validation(X, y):
    half = X.shape[0]/2
    X_val = X[half:,:]
    y_val = y[half:]
    X_test = X[:half,:]
    y_test = y[:half]
    return((X_val, y_val, X_test, y_test))