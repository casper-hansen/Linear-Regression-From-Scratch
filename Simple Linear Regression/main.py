from load_dataset import x_train, x_test, y_train, y_test
from simple_linear_regression import SimpleLinearRegression

# pick a single feature to estimate y
x_train = x_train['LSTAT'].values
x_test = x_test['LSTAT'].values
y_train = y_train.values
y_test = y_test.values

# fit to data
slr = SimpleLinearRegression()
slr.fit(x_train, y_train)

# make predictions and score
pred = slr.predict(x_test)
score = slr.r2_score(y_test, pred)
print(f'Final R^2 score: {score}')