from sklearn import datasets, model_selection
import numpy as np

X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

class BinaryLogisticRegression:

    def __init__(self, max_iterations=1000, learning_rate=0.001):
        self.max_iter = max_iterations
        self.lr = learning_rate
        self.w = None
        self.b = None

    def _initialise(self, n_features):
        pass

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        pass

    def _backwardpass(self):
        pass

    def _accuracy_score(self, prediction_labels, target_labels):
        correct_scores = prediction_labels == target_labels
        return np.sum(correct_scores) / len(prediction_labels)

    def score(self, X, y):

        if self.w == None or self.b == None:
            raise Exception('Need to fit the model to data.')

        if len(self.w) != X.shape[1]:
            raise Exception(f'Input data should be of shape (n_sample, {len(self.w)}).')

        predictions = self._sigmoid(X @ self.w + self.b)

        prediction_labels = predictions > 0.5

        return self._accuracy_score(prediction_labels, y)

    def predict(self, X):

        if self.w == None or self.b == None:
            raise Exception('Need to fit the model to data.')

        if len(self.w) != X.shape[1]:
            raise Exception(f'Input data should be of shape (n_sample, {len(self.w)}).')

        predictions = self._sigmoid(X @ self.w + self.b)

        prediction_labels = [x > 0.5 for x in predictions]

        return prediction_labels
        

