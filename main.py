from load_dataset import x_train, x_test, y_train, y_test
from multiple_linear_regression import MultipleLinearRegression
from sklearn.linear_model import LinearRegression

# # # # # # # # # # # # #
# Our Linear Regression #
# # # # # # # # # # # # #
mlr = MultipleLinearRegression()
mlr.fit(x_train, y_train)

# make predictions and score
pred = mlr.predict(x_test)

score = mlr.r2_score(y_test, pred)
print(f'Our Final R^2 score: {score}')

# # # # # # # # # # # # # # # # # #
# Scikit-Learn's Linear Regression #
# # # # # # # # # # # # # # # # # #
sk_mlr = LinearRegression()
sk_mlr.fit(x_train, y_train)
sk_score = sk_mlr.score(x_test, y_test)

print(f'Scikit-Learn\'s Final R^2 score: {sk_score}')