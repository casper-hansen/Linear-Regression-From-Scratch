import numpy as np

class MultipleLinearRegression():
    def __init__(self):
        self.coefficient = None
        self.intercept = None

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def r2_score(self, y_true, y_pred):
        pass

    def _estimate_coefficients(self, x, y):
        '''
            β = (X^T X)^-1 X^T y
        '''
        xT = x.transpose()

        inversed = np.linalg.inv( xT.dot(x) )
        coefficients = inversed.dot( xT ).dot(y)

        return coefficients

