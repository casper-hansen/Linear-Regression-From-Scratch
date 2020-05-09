import numpy as np

class MultipleLinearRegression():
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, x, y):
        x = self._transform_x(x)
        y = self._transform_y(y)

        betas = self._estimate_coefficients(x, y)
        
        self.intercept = betas[0]
        self.coefficients = betas[1:]

    def predict(self, x):
        '''
            y = b_0 + b_1*x + ... + b_i*x_i
        '''
        predictions = []
        for index, row in x.iterrows():
            values = row.values

            pred = np.multiply(values, self.coefficients)
            pred = np.add(pred, self.intercept)

            predictions.append(pred)

        return predictions

    def r2_score(self, y_true, y_pred):
        '''
            r2 = 1 - (rss/tss)
            rss = sum_{i=0}^{n} (y_i - y_hat)^2
            tss = sum_{i=0}^{n} (y_i - y_bar)^2
        '''
        y_average = np.average(y_true)

        residual_sum_of_squares = 0
        total_sum_of_squares = 0

        for i in range(len(y_true)):
            residual_sum_of_squares += (y_true[i] - y_pred[i])**2
            total_sum_of_squares += (y_true[i] - y_average)**2

        return 1 - (residual_sum_of_squares/total_sum_of_squares)

    def _transform_x(self, x):
        return x

    def _transform_y(self, y):
        return y

    def _estimate_coefficients(self, x, y):
        '''
            β = (X^T X)^-1 X^T y

            Estimates both the intercept and all coefficients.
        '''
        xT = x.transpose()

        inversed = np.linalg.inv( xT.dot(x) )
        coefficients = inversed.dot( xT ).dot(y)

        return coefficients

