# %%
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

os.chdir('../')
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """
    here we can conduct:
    1. Data Cleaning
    2. Feature Engineering
    3. Feature selection
    """
    return df

def load_data(data_path):
    df = pd.read_csv(data_path)
    df = preprocess_data(df)
    return df

def split_data(df):
    X = df.drop(columns=['Rating'])
    y = df['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def GridSearchRF(X_train, y_train, cv):
    pipe = Pipeline([('rf', RandomForestRegressor())])
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [10, 20, 30]
    }
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=cv)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    pipe = Pipeline(
        [('rf', RandomForestRegressor(n_estimators=best_params['rf__n_estimators'], max_depth=best_params['rf__max_depth']))]
    )

    return grid, pipe

def trainRF(X_train, y_train):
    pipe = Pipeline([('rf', RandomForestRegressor())])
    pipe.fit(X_train, y_train)
    return pipe

# def main():
#     print(f'current working directory: {os.getcwd()}')
#     data_path = 'data/zomato.csv'
#     df = load_data(data_path)
#     X_train, X_test, y_train, y_test = split_data(df)
#     grid, pipe = GridSearchRF(X_train, y_train, 5)
#     best_model = pipe
#     best_model.fit(X_train, y_train)
#     y_pred = pipe.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print('Mean Squared Error:', mse)
    
if __name__ == '__main__':
    print(f'current working directory: {os.getcwd()}')
    # %%
    print('Loading data...')
    data_path = 'data/task1_training_samples_features_display.csv'
    df = load_data(data_path)
    # %%
    X_train, X_test, y_train, y_test = split_data(df)
    # %%
    print('Grid Searching for best hyperparameters...')
    grid, pipe = GridSearchRF(X_train, y_train, 5)
    print('Best hyperparameters:', grid.best_params_)
    # %%
    print('Fitting the best model...')
    best_model = pipe
    best_model.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('Mean Squared Error:', mse)
    print('R2 Score:', r2)
    # %%
    print('Modifying the rating...')
    data_path = 'data/task1_training_samples_features_display.csv'
    df = load_data(data_path)
    df['Rating'].fillna(0, inplace=True)
    X_neg = df.drop(columns=['Rating'])
    y_neg = df['Rating']
    neg_pred = best_model.predict(X_neg)
    df['Rating_pred'] = neg_pred
    def modify_rating(rating, pred):
        if rating == 0:
            return pred
        else:
            return 0.6*rating + 0.4*pred
    df['Modified_Rating'] = df.apply(lambda x: modify_rating(x['Rating'], x['Rating_pred']), axis=1)
    # %%
    df['Rating']
    # %%
    df['Modified_Rating']
    # %%
    df['Rating_pred']



