from sklearn import datasets, model_selection

import matplotlib.pyplot as plt
import numpy as np
import random
import time

def fit_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print(f"Time taken to train the model: {round(time.time()-start, 6)} seconds")
    return wrapper

class BinaryLogisticRegression:
    '''
    This class is used to represent a binary logistic regression module.

    Attributes:
        max_iter (int): the maximum iterations to train the model.
        w (array): the weights of the model.
        b (float): the bias of the model.
        mean_X [array]: the mean values of each feature to be used in standardizing data.
        std_X [array]: the standard deviation values of each feature to be used in standardizing data.
    '''
    def __init__(self, max_iterations=500):
        self.max_iter = max_iterations
        self.w = []
        self.b = None
        self.mean_X = []
        self.std_X = []

    def _initialise(self, n_features):
        """ Initialises the random weights and bias of the model with the correct dimensions

        Args:
            n_features (int): the number of features in the data.
        """
        self.w = np.random.randn(n_features)
        self.b = np.random.rand()

    def _standarize(self, X):
        """ Standardizes the data. 

        Args:
            X (array): the data to be standardized.

        Returns:
            standardized_X: the standardized data
        """
        if len(self.mean_X) == 0:
            self.mean_X = np.mean(X, axis=0)
            self.std_X = np.std(X, axis=0)

        standardized_X = (X - self.mean_X) / self.std_X
        return standardized_X

    def _sigmoid(self, z):
        """ The sigmoid function.

        Args:
            z (float or array): the input to apply the sigmoid function to.

        Returns:
            output: the output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def _calculate_loss(self, X, y):
        """ Calculates the Binary Cross Entropy loss with logits.

        Args:
            X (array): the matrix of feature data.
            y (array): the array of labels.

        Returns:
            average_loss: the average BCE loss.
        """
        m = len(y)
        logit = X @ self.w + self.b

        average_loss = np.sum( np.max(np.vstack((logit, m*[0])), axis=0) - logit*y + np.log(1 + np.exp(-abs(logit))) ) / m

        return average_loss

    def _step(self, X, y, lr):
        """ Updates the parameters of the model using gradient descent.

        Args:
            X (array): the feature data.
            y (array): the label data.
            lr (float): the learning rate.
        """
        logit = X @ self.w + self.b

        dLdw = X.T @ (self._sigmoid(logit) - y)
        dLdb = np.sum(self._sigmoid(logit) - y)

        self.w = self.w - lr*dLdw
        self.b = self.b - lr*dLdb

    def _get_minibatch(self, X, y, size):
        """ Returns a minibatch of the data to be used in minibatch stochastic gradient descent. 

        Args:
            X (array): the feature data.
            y (array): the label data.
            size (int): the size of the minibatch

        Returns:
            new_X, new_y: the minibatch data.
        """
        x_columns = X.shape[1]
        data = np.hstack((X, y))
        np.random.shuffle(data)

        new_X, new_y = data[:size,:x_columns], data[:size,x_columns:]

        return new_X, new_y

    @fit_timer
    def fit(self, X, y, lr=0.001, minibatch_size='all_data', verbose=True):
        """ Fits the model parameters to the input data. 

        Args:
            X (array): the feature data.
            y (array): the label data.
            lr (float): the learning rate.
            minibatch_size (int): the size of the batches to be used in minibatch gradient descent.
            verbose (bool): indicates whether an update on the loss should be printed every 100 iterations.
        """
        n_features = X.shape[1]
        m = len(y)
        self._initialise(n_features)

        X = self._standarize(X)
        all_losses = []

        for i in range(self.max_iter):

            if type(minibatch_size) != str and minibatch_size < m:
                X, y = self._get_minibatch(X, y, minibatch_size)

            loss = self._calculate_loss(X, y)
            all_losses.append(loss)

            if verbose == True:
                if i % 100 == 0:
                    print(f'Iteration {i}, Loss = {loss}')

            self._step(X, y, lr)

            if len(all_losses) > 1:
                if (all_losses[-2] - all_losses[-1])/all_losses[-2] < 0.00001:
                    break

        self._losses = all_losses

    def _accuracy_score(self, prediction_labels, target_labels):
        """ Returns the accuracy score for predicted labels. 

        Args:
            prediction_labels (array): the predicted labels.
            target_labels (array): the ground truth labels.

        Returns:
            accuracy score: A score of how accurate the labels are -> [0,1]
        """
        correct_scores = prediction_labels == target_labels
        return np.sum(correct_scores) / len(prediction_labels)

    def score(self, X, y):
        """ Caclulates the accuracy score for some input feature data. 

        Args:
            X (array): the feature data.
            y (array): the label data.

        Returns:
            accuracy score: A score of how accurate the labels are -> [0,1]
        """
        if len(self.w) == 0:
            raise Exception('Need to fit the model to data.')

        if len(self.w) != X.shape[1]:
            raise Exception(f'Input data should be of shape (n_sample, {len(self.w)}).')

        X = self._standarize(X)

        predictions = self._sigmoid(X @ self.w + self.b)

        prediction_labels = predictions > 0.5

        return self._accuracy_score(prediction_labels, y)

    def predict(self, X):
        """ Returns the predicted labels for input feature data 

        Args:
            X (array): the feature data.

        Returns:
            prediction_labels: the predicted labels
        """
        if len(self.w) == 0:
            raise Exception('Need to fit the model to data.')

        if len(self.w) != X.shape[1]:
            raise Exception(f'Input data should be of shape (n_sample, {len(self.w)}).')

        X = self._standarize(X)

        predictions = self._sigmoid(X @ self.w + self.b)

        prediction_labels = [x > 0.5 for x in predictions]

        return prediction_labels

class MultinomialLogisticRegression:
    '''
    This class is used to represent a multinomial logistic regression module.

    Attributes:
        max_iter (int): the maximum iterations to train the model.
        w (array): the weights of the model.
        b (float): the bias of the model.
        mean_X [array]: the mean values of each feature to be used in standardizing data.
        std_X [array]: the standard deviation values of each feature to be used in standardizing data.
    '''
    def __init__(self, max_iterations=5000):
        self.max_iter = max_iterations
        self.w = []
        self.b = None
        self.mean_X = []
        self.std_X = []

    def _initialise(self, n_features, n_classes):
        """ Initialises the random weights and bias of the model with the correct dimensions

        Args:
            n_features (int): the number of features in the data.
            n_classes (int): the number of classes in the label data.
        """
        self.w = np.random.randn(n_features,n_classes)
        self.b = np.random.randn(1,n_classes)

    def _standarize(self, X):
        """ Standardizes the data. 

        Args:
            X (array): the data to be standardized.

        Returns:
            standardized_X: the standardized data
        """
        if len(self.mean_X) == 0:
            self.mean_X = np.mean(X, axis=0)
            self.std_X = np.std(X, axis=0)

        standardized_X = (X - self.mean_X) / self.std_X
        return standardized_X

    def _softmax(self, z):
        """ The sigmoid function.

        Args:
            z (float or array): the input to apply the sigmoid function to.

        Returns:
            output: the output of the softmax function with shape (m,k) where m is the samples and k is the num classes.
        """
        exponentials = np.exp(z)
        exp_sums = np.sum(exponentials, axis=1)
        softmax = exponentials / exp_sums[:,None]
        return softmax

    def _calculate_loss(self, X, y):
        """ Calculates the Cross Entropy loss.

        Args:
            X (array): the matrix of feature data.
            y (array): the array of labels.

        Returns:
            average_loss: the average CE loss.
        """
        m = y.shape[0]
        logit = X @ self.w + self.b
        softmax_output = self._softmax(logit)
        total_loss = np.sum(y * -np.log(softmax_output))

        average_loss = total_loss/ m

        return average_loss

    def _step(self, X, y, lr):
        """ Updates the parameters of the model using gradient descent.

        Args:
            X (array): the feature data.
            y (array): the label data.
            lr (float): the learning rate.
        """
        logit = X @ self.w + self.b
        softmax_output = self._softmax(logit)

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = y.shape[1]

        diff = y - softmax_output
        dLdb = np.sum(diff, axis=0)
        dLdw = X.T @ diff

        dLdb *= -1/n_samples
        dLdw *= -1/n_samples

        self.w = self.w - lr*dLdw
        self.b = self.b - lr*dLdb

    def _get_minibatch(self, X, y, size):
        """ Returns a minibatch of the data to be used in minibatch stochastic gradient descent. 

        Args:
            X (array): the feature data.
            y (array): the label data.
            size (int): the size of the minibatch

        Returns:
            new_X, new_y: the minibatch data.
        """
        x_columns = X.shape[1]
        data = np.hstack((X, y))
        np.random.shuffle(data)

        new_X, new_y = data[:size,:x_columns], data[:size,x_columns:]

        return new_X, new_y

    def _one_hot_encoder(self, labels, max_Labels = None):
        """ Takes in label encoded label data and one hot encoded labels. 

        Args:
            labels (array): the label data.
            max_labels (int): the number of classes.

        Returns:
            ohe labels: the one hot encoded labels.
        """
        if max_Labels == None:
            max_Labels = np.max(labels) + 1
        return np.eye(max_Labels)[labels]

    def _to_labels(self, one_hot):
        """ Takes in one hot encoded labels and returns it as label encoded. 

        Args:
            one_hot (array): the one hot encoded label data.

        Returns:
            labels: the label encoded labels.
        """
        return np.argmax(one_hot, axis=-1)

    @fit_timer
    def fit(self, X, y, lr=0.001, minibatch_size='all_data', verbose=True):
        """ Fits the model parameters to the input data. 

        Args:
            X (array): the feature data.
            y (array): the label data.
            lr (float): the learning rate.
            minibatch_size (int): the size of the batches to be used in minibatch gradient descent.
            verbose (bool): indicates whether an update on the loss should be printed every 100 iterations.
        """
        n_features = X.shape[1]
        m = len(y)
        y = self._one_hot_encoder(y)
        n_classes = y.shape[1]
        self._initialise(n_features, n_classes)

        X = self._standarize(X)
        all_losses = []

        for i in range(self.max_iter):

            if type(minibatch_size) != str and minibatch_size < m:
                X_batch, y_batch = self._get_minibatch(X, y, minibatch_size)
            else:
                X_batch, y_batch = X, y

            loss = self._calculate_loss(X, y)
            all_losses.append(loss)

            if verbose == True:
                if i % 100 == 0:
                    print(f'Iteration {i}, Loss = {loss}')

            self._step(X_batch, y_batch, lr)

            if len(all_losses) > 1:
                if (all_losses[-2] - all_losses[-1])/all_losses[-2] < 0.00001:
                    break

        self._losses = all_losses

    def _accuracy_score(self, prediction_labels, target_labels):
        """ Returns the accuracy score for predicted labels. 

        Args:
            prediction_labels (array): the predicted labels.
            target_labels (array): the ground truth labels.

        Returns:
            accuracy score: A score of how accurate the labels are -> [0,1]
        """
        correct_scores = prediction_labels == target_labels
        return np.sum(correct_scores) / len(prediction_labels)

    def score(self, X, y):
        """ Caclulates the accuracy score for some input feature data. 

        Args:
            X (array): the feature data.
            y (array): the label data.

        Returns:
            accuracy score: A score of how accurate the labels are -> [0,1]
        """
        if len(self.w) == 0:
            raise Exception('Need to fit the model to data.')

        if len(self.w) != X.shape[1]:
            raise Exception(f'Input data should be of shape (n_sample, {len(self.w)}).')

        X = self._standarize(X)

        predictions = np.argmax(self._softmax(X @ self.w + self.b), axis=1)

        return self._accuracy_score(predictions, y)

    def predict(self, X):
        """ Returns the predicted labels for input feature data 

        Args:
            X (array): the feature data.

        Returns:
            prediction_labels: the predicted labels
        """
        if len(self.w) == 0:
            raise Exception('Need to fit the model to data.')

        if len(self.w) != X.shape[1]:
            raise Exception(f'Input data should be of shape (n_sample, {len(self.w)}).')

        X = self._standarize(X)

        predictions = np.argmax(self._softmax(X @ self.w + self.b), axis=1)

        return predictions
        
if __name__ == "__main__":

    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

    model = MultinomialLogisticRegression(max_iterations=10000)
    model.fit(X_train, y_train, verbose=False)

    print(model.score(X_test, y_test))

    plt.figure()
    plt.plot(model._losses)
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.title('Cross Entropy loss vs Iterations')
    plt.show()