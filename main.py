from sklearn.datasets import load_boston
from linear_regression import LinearRegression

# load dataset
boston = load_boston()
x = boston.data
y = boston.target

# instantiate linear regression class
lr = LinearRegression()
lr.fit()