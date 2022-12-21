from sklearn import datasets, model_selection

X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

class BinaryLogisticRegression:

    def __init__(self, max_iterations=1000, learning_rate=0.001):
        self.max_iter = max_iterations
        self.lr = learning_rate

    def _initialise(self, n_features):
        pass
    
    def fit(self, X, y):
        pass

    def _backwardpass(self):
        pass

    def score(self, X, y):
        pass

    def predict(self, X):
        pass

