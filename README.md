# Binary Logistic regression from scratch

In the main.py file I have created a class for a Binary Logistic Regression model from scratch. The breast cancer dataset from sklearn has been used to test the model and the loss curve and accuracy score for the test set are displayed.

The loss used in this model is the Binary Cross Entropy loss with logits which is shown below:

`BCE = max(0,z) - z*y + ln( 1 + exp( -abs(z) ) )`

Where `z` is the logit, which in the case of logistic regression is `z = XW + b`, and `y` is the true class label.

The accuracy score for the model is roughly 0.97 on test data.

I have also included a decorator to time the fit method of the class to gauge how fast the training is.

# Multinomial Logistic regression from scratch

I have added a class that is able to do multinomial logistic regression (multiclass classification). The iris dataset from sklearn was used to test the model and it gave an accuracy score of ~ 0.8.

The loss used for this was the cross entropy loss.

The class can be used with minibatch gradient descent for improved efficiency. Since their is no scale to the label data in the iris dataset, a one hot encoder function was included to be able to use the label data as one hot encoded.