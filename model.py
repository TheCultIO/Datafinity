import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def drop_columns_with_high_null(df, threshold=0.75):
    null_percentages = df.isnull().mean()
    columns_to_drop = null_percentages[null_percentages > threshold].index
    df = df.drop(columns=columns_to_drop)
    return df


def segregate_columns(df):
    categorical_cols = df.select_dtypes(include='object').columns
    numerical_cols = df.select_dtypes(include=['int', 'float']).columns
    return categorical_cols, numerical_cols


def predict(df1):
    df = pd.read_csv('compressed data.csv')

    df = df.drop_duplicates()
    df = drop_columns_with_high_null(df, threshold=0.75)
    df.drop(["id", "city", "country", "county", "countyFIPS"], axis=1, inplace=True)

    categorical_cols, numerical_cols = segregate_columns(df)

    for i in numerical_cols:
        df[i].fillna(df[i].mean(), inplace=True)
    for i in categorical_cols:
        df[i].fillna(df[i].mode()[0], inplace=True)

    columns_to_drop = ['mostRecentPriceDomain', 'prices.currency', 'prices.isSale']  # List of column names to drop
    df.drop(columns=columns_to_drop, inplace=True)

    for col in categorical_cols:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])

    target_column = 'prices.pricePerSquareFoot'
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.drop(target_column, axis=1))  # Drop the target column before scaling

    # Step 4: Feature Selection
    X = df_scaled
    y = df[target_column]

    model = RandomForestRegressor()
    model.fit(X, y)

    # New dataframe
    df1 = df1.drop_duplicates()
    df2 = df1.copy()
    df1 = drop_columns_with_high_null(df1, threshold=0.75)
    df1.drop(["id", "city", "country", "county", "countyFIPS"], axis=1, inplace=True)

    categorical_cols1, numerical_cols1 = segregate_columns(df1)

    for i in numerical_cols1:
        df1[i].fillna(df1[i].mean(), inplace=True)
    for i in categorical_cols1:
         df1[i].fillna(df1[i].mode()[0], inplace=True)

    columns_to_drop = ['mostRecentPriceDomain', 'prices.currency', 'prices.isSale']  # List of column names to drop
    df1.drop(columns=columns_to_drop, inplace=True)

    for col in categorical_cols1:
        label_encoder = LabelEncoder()
        df1[col] = label_encoder.fit_transform(df1[col])

    scaler = StandardScaler()
    df_scaled1 = scaler.fit_transform(df1.drop(target_column, axis=1))

    y_pred = model.predict(df_scaled1)

    df2['prices.pricePerSquareFoot'] = y_pred

    return df2
