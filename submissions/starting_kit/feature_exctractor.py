
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0


def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())

    d = d[columns]
    return d


class FeatureExtractor():
    def __init__(self):
        pass
    
    #def __init__(self,attribute_names):
       # self.attribute_names = attribute_names
        
                
        
    def fit(self, X_df, y=None):
        global column_dummies
        if y is not None:
            column_dummies = pd.concat(
            [X_df.get(['yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'monthOfRegistration']),
             pd.get_dummies(X_df.seller, prefix = 'Size', drop_first=True),
             #pd.get_dummies(X_df.offerType, prefix='Auction', drop_first=True),
             pd.get_dummies(X_df.vehicleType, prefix='Color', drop_first=True),
             pd.get_dummies(X_df.fuelType, prefix='Transmission', drop_first=True),
             pd.get_dummies(X_df.brand, prefix='Nationality', drop_first=True),
             pd.get_dummies(X_df.notRepairedDamage, prefix='notRepairedDamage', drop_first=True),
             
             ],
            axis=1).columns
        return self
        return self
    
    
    def transform(self, X_df):
        #print(column_dummies)
        X_df_new = pd.concat(
            [X_df.get(['yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'monthOfRegistration']),
             pd.get_dummies(X_df.seller, prefix = 'seller', drop_first=True),
             #pd.get_dummies(X_df.offerType, prefix='Auction', drop_first=True),
             pd.get_dummies(X_df.vehicleType, prefix='vehicleType', drop_first=True),
             pd.get_dummies(X_df.fuelType, prefix='fuelType', drop_first=True),
             pd.get_dummies(X_df.brand, prefix='brand', drop_first=True),
             pd.get_dummies(X_df.notRepairedDamage, prefix='notRepairedDamage', drop_first=True),
             
             ],
            axis=1)
        #X_df_new = X_df_new.fillna(-1)
        
        X_df_new = fix_columns(X_df_new, column_dummies)
        
        scaler = StandardScaler()

        #X_df_new[['yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'monthOfRegistration']] = scaler.fit_transform(X_df_new [['yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'monthOfRegistration']])
        
        X_df_new= X_df_new.as_matrix()
        return X_df_new