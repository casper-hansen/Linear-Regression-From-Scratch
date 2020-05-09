import numpy as np

class SimpleLinearRegression():
    def __init__(self):
        self.coefficient = None
        self.intercept = None

    def fit(self, x, y):
        self.coefficient = self._coefficient_estimate(x, y)
        self.intercept = self._compute_intercept(x, y)

    def predict(self, x):
        x_times_coeff = np.multiply(x, self.coefficient)
        return np.add(x_times_coeff, self.intercept)

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

    def _compute_intercept(self, x, y):
        '''
            intercept = y_bar - coefficient*x_bar
            
            WHERE:  y_bar = average(y),
                    x_bar = average(x),
                    coefficient = coefficient already estimated
        '''
        # for each feature, find the average
        x_average = np.average(x)

        # multiply the coefficient and the average of the x values
        mul = self.coefficient*x_average

        return np.average(y) - mul

    def _coefficient_estimate(self, x, y):
        '''
            coefficient_{x,y} = ∑_{i=0}^{n} (x_i - x_bar) * (y_i - y_bar)
                                _________________________________________
                                ∑_{i=0}^{n} (x_i - x_bar)^2
        '''
        numerator = 0
        denominator = 0

        for i in range(len(x)):
            x_value = x[i]
            y_value = y[i]
            x_average = np.average(x)
            y_average = np.average(y)

            numerator += (x_value - x_average) * (y_value - y_average)
            denominator += (x_value - x_average)**2

        return numerator / denominator