from sklearn.datasets import load_boston
import pandas as pd

def sklearn_to_df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    X = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')

    return X, y

x, y = sklearn_to_df(load_boston())