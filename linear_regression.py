import numpy as np

class LinearRegression():
    def __init__(self):
        self.intercept = []
        self.coefficients = []
        self.noise = []

    def fit(self, x, y):
        # coeffs = [self._coefficient_estimate(x_feature, y) for x_feature in x]
        # intercept = self._compute_intercept(coeffs, x, y)
        # noise = ?
        pass

    def predict(self, x):
        # self._multiply_coeffs(x)
        pass

    def score(self):
        pass

    def _multiply_coeffs(self, x):
        mul = np.multiply(x, self.coeffs)
        return sum(mul)

    def _compute_intercept(self, x, y):
        '''
            intercept = y_bar - coeffs*x_bar
            
            WHERE:  y_bar = average(y), 
                    x_bar = average(x),
                    coeffs = coefficients already estimated
        '''
        # for each feature, find the average
        averages_of_x = [np.average(x_feature) for x_feature in x]

        # multiple the averages by the coefficients and sum
        coeffs_times_x = self._multiply_coeffs(averages_of_x)

        return np.average(y) - coeffs_times_x

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