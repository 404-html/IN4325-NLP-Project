from sklearn import svm
from sklearn.metrics import f1_score

def tune_params(X_train, y_train, X_val, y_val, verbose):
    best_f1 = 0
    best_params = None

    kernels = ['linear', 'poly', 'rbf']
    degrees = [2,3,4]
    gammas = ['auto', 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] # Only for non-linears. Higher gammas tend to over-fit
    Cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] # Penalty for classifying wrongly. Higher Cs tend to over-fit
    functions = ['ovo', 'ovr']

    for k in kernels:
        for f in functions:
            for C in Cs:
                if k=='linear':
                    clf = svm.SVC(kernel=k,C=C,decision_function_shape=f)
                    clf.fit(X_train, y_train)
                    y_predicted = clf.predict(X_val)
                    f1 = f1_score(y_val, y_predicted, average='micro')
                    if f1>=best_f1:
                        best_f1 = f1
                        best_params = clf.get_params()
                    if verbose:
                        print(clf.get_params)
                        print(f1)
                        plot_confusion_matrix(y_val, y_predicted, np.array(('0', '1', '2')))
                else:
                    for g in gammas:
                        if k=='poly':
                            for d in degrees:
                                clf = svm.SVC(kernel=k,gamma=g,degree=d,C=C,decision_function_shape=f)
                                clf.fit(X_train, y_train)
                                y_predicted = clf.predict(X_val)
                                f1 = f1_score(y_val, y_predicted, average='micro')
                                if f1>best_f1:
                                    best_f1 = f1
                                    best_params = clf.get_params()
                                if verbose:
                                    print(clf.get_params)
                                    print(f1)
                                    plot_confusion_matrix(y_val, y_predicted, np.array(('0', '1', '2')))
                        else:
                            clf = svm.SVC(kernel=k,gamma=g,C=C,decision_function_shape=f)
                            clf.fit(X_train, y_train)
                            y_predicted = clf.predict(X_val)
                            f1 = f1_score(y_val, y_predicted, average='micro')
                            if f1>best_f1:
                                best_f1 = f1
                                best_params = clf.get_params()
                            if verbose:
                                print(clf.get_params)
                                print(f1)
                                plot_confusion_matrix(y_val, y_predicted, np.array(('0', '1', '2')))
    return best_params