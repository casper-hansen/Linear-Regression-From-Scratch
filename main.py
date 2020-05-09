from load_dataset import x_train, x_test, y_train, y_test
from multiple_linear_regression import MultipleLinearRegression

# fit to data
mlr = MultipleLinearRegression()
mlr.fit(x_train, y_train)

# make predictions and score
pred = mlr.predict(x_test)
score = mlr.r2_score(y_test, pred)
print(f'Final R^2 score: {score}')