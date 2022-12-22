from sklearn import datasets, model_selection
import numpy as np
import matplotlib.pyplot as plt

class BinaryLogisticRegression:

    def __init__(self, max_iterations=500):
        self.max_iter = max_iterations
        self.w = []
        self.b = None
        self.mean_X = []
        self.std_X = []

    def _initialise(self, n_features):
        self.w = np.random.randn(n_features)
        self.b = np.random.rand()

    def _standarize(self, X):

        if len(self.mean_X) == 0:
            self.mean_X = np.mean(X, axis=0)
            self.std_X = np.std(X, axis=0)

        standardized_X = (X - self.mean_X) / self.std_X
        return standardized_X

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _calculate_loss(self, X, y):
        
        total_loss = 0
        m = len(y)

        for i in range(m):
            logit = np.dot(X[i,:], self.w) + self.b
            loss = np.max([0, logit]) - y[i]*logit + np.log(1 + np.exp(-abs(logit)))
            total_loss += loss

        average_loss = total_loss / m

        return average_loss

    def _step(self, X, y, lr):

        logit = X @ self.w + self.b

        dLdw = X.T @ (self._sigmoid(logit) - y)
        dLdb = np.sum(self._sigmoid(logit) - y)

        self.w = self.w - lr*dLdw
        self.b = self.b - lr*dLdb

    def fit(self, X, y, lr=0.001):

        n_features = X.shape[1]
        self._initialise(n_features)

        X = self._standarize(X)
        all_losses = []

        for i in range(self.max_iter):
            loss = self._calculate_loss(X, y)
            all_losses.append(loss)

            if i % 100 == 0:
                print(f'Iteration {i}, Loss = {loss}')

            self._step(X, y, lr)

            if len(all_losses) > 1:
                if (all_losses[-2] - all_losses[-1])/all_losses[-2] < 0.00001:
                    break

        return all_losses

    def _accuracy_score(self, prediction_labels, target_labels):
        correct_scores = prediction_labels == target_labels
        return np.sum(correct_scores) / len(prediction_labels)

    def score(self, X, y):
        if len(self.w) == 0:
            raise Exception('Need to fit the model to data.')

        if len(self.w) != X.shape[1]:
            raise Exception(f'Input data should be of shape (n_sample, {len(self.w)}).')

        X = self._standarize(X)

        predictions = self._sigmoid(X @ self.w + self.b)

        prediction_labels = predictions > 0.5

        return self._accuracy_score(prediction_labels, y)

    def predict(self, X):
        if len(self.w) == 0:
            raise Exception('Need to fit the model to data.')

        if len(self.w) != X.shape[1]:
            raise Exception(f'Input data should be of shape (n_sample, {len(self.w)}).')

        X = self._standarize(X)

        predictions = self._sigmoid(X @ self.w + self.b)

        prediction_labels = [x > 0.5 for x in predictions]

        return prediction_labels
        

if __name__ == "__main__":

    X, y = datasets.load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    model = BinaryLogisticRegression()
    losses = model.fit(X_train, y_train)
    
    print(model.score(X_test, y_test))

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.title('Binary Cross Entropy loss with logits vs Iterations')
    plt.show()