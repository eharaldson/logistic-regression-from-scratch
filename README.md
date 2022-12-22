# Binary Logistic regression from scratch

In the main.py file I have created a class for a Binary Logistic Regression model from scratch. The breast cancer dataset from sklearn has been used to test the model and the loss curve and accuracy score for the test set are displayed.

The loss used in this model is the Binary Cross Entropy loss with logits which is shown below:

`BCE = max(0,z) - z*y + ln( 1 + exp( -abs(z) ) )`

Where `z` is the logit, which in the case of logistic regression is `z = XW + b`, and `y` is the true class label.

The accuracy score for the model is roughly 0.97 on test data.

I have also included a decorator to time the fit method of the class to gauge how fast the training is.