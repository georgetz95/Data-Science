from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
import pandas as pd
import numpy as np

num_cols = ['creditLimit',
 'availableMoney',
 'transactionAmount',
 'cardCVV',
 'enteredCVV',
 'cardLast4Digits',
 'currentBalance',
 'cardPresent',
 'expirationDateKeyInMatch',
 'trans_month',
 'trans_day_addr_change_diff',
 'trans_day_open_date_diff', 'exp_month', 'exp_year']

categorical_cols = ['brandName',
 'acqCountry',
 'merchantCountryCode',
 'posEntryMode',
 'posConditionCode',
 'merchantCategoryCode',
 'transactionType',
 'trans_day_name',
 'trans_day_of_week']

class DataClean(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent')
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            if any(X[col] == ''):
                X.loc[X[col] == '', col] = np.nan
            if X[col].dtype == bool:
                X.loc[:,col] = X[col].astype(int)
                
        X = X.drop(columns=['echoBuffer', 'merchantCity', 'merchantState',
                            'merchantZip', 'posOnPremises', 'recurringAuthInd', 'customerId', 'accountNumber'])
                
        X['acqCountry'] = X.groupby('merchantCountryCode')['acqCountry'].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x.iloc[0])
        missing_features = X.columns[X.isna().any()].tolist()
        # X[missing_features] = pd.DataFrame(self.imputer.fit_transform(X[missing_features]), columns=missing_features)
        X[missing_features] = self.imputer.fit_transform(X[missing_features])
        # X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        return X
    
class FeatureExtraction(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['transactionDateTime'] = pd.to_datetime(X['transactionDateTime'], errors='coerce')
        X['accountOpenDate'] = pd.to_datetime(X['accountOpenDate'], errors='coerce')
        X['dateOfLastAddressChange'] = pd.to_datetime(X['dateOfLastAddressChange'], errors='coerce')
        X['currentExpDate'] = pd.to_datetime(X['currentExpDate'], errors='coerce', format='%m/%Y')
        
        X['trans_day_name'] = X.transactionDateTime.apply(lambda x: datetime.strftime(x, '%A'))
        X['trans_day_of_week'] = X.transactionDateTime.apply(lambda x: datetime.strftime(x, '%A'))
        X['trans_month'] = X.transactionDateTime.apply(lambda x: int(datetime.strftime(x, '%m')))
        X['trans_day_addr_change_diff'] = (X.transactionDateTime - X.dateOfLastAddressChange).dt.days
        X['trans_day_open_date_diff'] = (X.transactionDateTime - X.accountOpenDate).dt.days
        X['exp_month'] = X.currentExpDate.apply(lambda x:int(datetime.strftime(x, '%m')))
        X['exp_year'] = X.currentExpDate.apply(lambda x:int(datetime.strftime(x, '%Y')))
        X['matchingCVV'] = X.cardCVV == X.enteredCVV
        X['brandName'] = X['merchantName'].apply(lambda x: x.split('#')[0].strip())
        X = X.drop(columns=['merchantName','currentExpDate', 'transactionDateTime', 'accountOpenDate', 'dateOfLastAddressChange'])
        
        
        return X
    
col_transformer = ColumnTransformer([('standard_scaler', StandardScaler(), num_cols),
                                    ('onehot_encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), categorical_cols)
                                   ])

pipeline = Pipeline([
    ('DataClean', DataClean()),
    ('FeatureExtraction', FeatureExtraction()),
    ('col_transformer', col_transformer)
])