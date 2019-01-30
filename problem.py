
import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
import numpy as np

problem_title = 'Cars price'
_target_column_names = 'price'
_ignore_column_names = ["postalCode", "offerType" ]
#_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_regression(
    label_names=[_target_column_names])
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()




# New Error
# To penalize more if one did not predict a 'risk zone'


score_types = [
    rw.score_types.RMSE(name='rmse'),
    #rw.score_types.Accuracy(name='acc'),
    #rw.score_types.NegativeLogLikelihood(name='nll'),
]


def get_cv(X, y):
    bins     = np.linspace(0, max(y), 3)
    y_binned = np.digitize(y, bins)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y_binned)
    #cv = ShuffleSplit(n_splits=8, test_size=0.2)
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y_binned)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    # filter outliers 
    data = data[
    (data["yearOfRegistration"].between(1945, 2017, inclusive=True)) &
    (data["powerPS"].between(100, 500, inclusive=True)) &
    (data["price"].between(100, 200000, inclusive=True))
        ]
    y_array = data[_target_column_names].values
    X_df = data.drop([_target_column_names] + _ignore_column_names, axis=1)
    X_df = X_df.reset_index().drop(['index' ], axis=1)
    return clean_and_transform(X_df), y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)

def clean_and_transform(X):
    data = X.copy()
    
    # Replace the NaN-Values
    data['namelen'] = [min(70, len(n)) for n in data['name']]
    

    data['vehicleType'].fillna(value='blank', inplace=True)
    data['gearbox'].fillna(value='blank', inplace=True)
    data['model'].fillna(value='blank', inplace=True)
    data['fuelType'].fillna(value='blank', inplace=True)
    data['notRepairedDamage'].fillna(value='blank', inplace=True)

    for col in data[:-3]:
        if data[col].dtype == "object":
            data[col] = data[col].astype('category')

    # Assign codes to categorical attributes instead of strings        
    cat_columns = data.select_dtypes(['category']).columns

    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    data = data.drop(['name' ], axis=1)
    
    

    return data

def drop_column(X, columns_to_drop=["dateCrawled", "abtest", "dateCreated", "lastSeen"]):
    data = X.copy()
    data = data.drop(columns_to_drop, axis=1)
    return data

