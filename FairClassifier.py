import numpy as np

from sklearn import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize


class FairClassifier(BaseEstimator, ClassifierMixin):
    """ Classifier fitting the scikit-learn guidelines.
    Implements the BaseEstimator and the RegressorMixin modules from sklearn.
    
    Methods
    -------
    __init__(self, proxy_correction=True): Constructor
    
    fit(self, X, y): fits the Classifier to a data set X and output y
    """
    
    def __init__(self, proxy_correction=True):
        self._weights_ = []
        self._initialized_ = False
        self.proxy_correction = proxy_correction
            
    def fit(self, X, y):
        
        if proxy_discriminiation:
            X = cancel_proxy_discrimination(X, y)
        
        X_intercepts = FairClassifier.add_intercept(X)
        
        self._weights_ = minimize(fun=FairClassifier.logistic_loss,
                                x0=np.random.rand(x.shape[1]),
                                args=(X_intercepts,y),
                                options={"maxiter": 10000})
        
        self._initialized_ = True
        
        return self
    
    def predict(self, X):
        """Predict labels for X.
        
        Parameters
        ----------
            X
        
        Returns
        -------
            y_predicted: predicted labels
        """
        if not self._initialized:
            raise ValueError("Model not initialized. You have to run 'fit' first.")
           
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy ndarray.")
            
        y_predicted = np.sign(np.dot(FairClassifier.add_intercept(dataset), self._weights_))
        
        y_predicted[(y_predicted == -1)] = 0
            
            
         
        return self
    
    def cancel_proxy_discrimination(X, y):
        return X
    
    @staticmethod
    def logistic_loss(w, X, y, return_arr=False):
        """Numpy implementation of logistic loss.
        Uses the 
        
        This function is used from fairensics by nikikilbertus. 
        He again used it from fair_classification and the scikit-learn source code.
        
        Source code at:
        https://github.com/nikikilbertus/fairensics/blob/master/fairensics/methods/utils.py
        https://github.com/mbilalzafar/fair-classification/blob/master/fair_classification/loss_funcs.py
        
        Parameters
        ----------
            w (np.ndarray): 1D, the weight matrix with shape (n_features,).
            X (np.ndarray): 2D, the features with shape (n_samples, n_features).
            y (np.ndarray): 1D, the true labels with shape (n_samples,).
            return_arr (bool): if true, an array is returned otherwise the sum of the array
            
        Returns
        -------
            (float or list(float)): the loss.
        """

        yz = y * np.dot(X, w)
        # Logistic loss is the negative of the log of the logistic function.
        if return_arr:
            return -FairClassifier.log_logistic(yz)

        return -np.sum(FairClassifier.log_logistic(yz))
        
    @staticmethod
    def log_logistic(X):
        
        """Log_logistic taken from fairensics by nikikilbertus. He again used scikit-learn
        as an inspiration. Both Links below.
        
        Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
        
        The implementation splits positive and negative values and thus is numerically stable:
        
            -log(1 + exp(-x_i))     if x_i > 0
            x_i - log(1 + exp(x_i)) if x_i <= 0
        
        Source code at:
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
        https://github.com/nikikilbertus/fairensics/blob/master/fairensics/methods/utils.py
        
        Parameters
        ----------
            X (array-like): shape (M, N) 
                            Argument to the logistic function
            
        Returns
        -------
            out (np.ndarray): shape (M, N) 
                              Log of the logistic function at every point in x
        """
        
        if X.ndim > 1:
            raise Exception("Array of samples cannot be more than 1-D!")
        out = np.empty_like(X)  # same dimensions and data types

        idx = X > 0
        out[idx] = -np.log(1.0 + np.exp(-X[idx]))
        out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
        return out 
    
    @staticmethod
    def add_intercept(X):
    """Adds intercept (column of ones) to X.
    Taken from fairensics. Soruce code:
    https://github.com/nikikilbertus/fairensics/blob/master/fairensics/methods/utils.py
    """
    m, _ = X.shape
    intercept = np.ones(m).reshape(m, 1)  # the constant b
    return np.concatenate((intercept, X), axis=1)