#!/bin/python

def train_classifier(X, y):
    """Train a classifier using the given training data.
        
        Trains logistic regression on the input data with default parameters.
        """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C': [0.01, 0.05, 0.07, 0.08, 0.1, 0.12, 0.15, 0.25, 0.5, 0.65, 0.80, 1,2,3,4,  5, 10, 100, 200, 250, 400, 500, 1000]}
    grid = GridSearchCV(LogisticRegression(random_state=0, solver='lbfgs',class_weight = 'balanced', max_iter=10000), param_grid, cv=5)
    grid.fit(X, y)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    cls = grid.best_estimator_
    #cls = LogisticRegression(C=0.15, class_weight='balanced', dual=False,
    #      fit_intercept=True, intercept_scaling=1, max_iter=10000,
    #      multi_class='warn', n_jobs=None, penalty='l2', random_state=0,
    #      solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
    #cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000)
    cls.fit(X, y)
    return cls

def evaluate(X, yt, cls, name='data'):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print("  Accuracy on %s  is: %s" % (name, acc))
