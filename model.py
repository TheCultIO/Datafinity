import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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


def train_model():
    df = pd.read_csv('compressed data.csv')

    df = df.drop_duplicates()
    df = drop_columns_with_high_null(df, threshold=0.75)
    df.drop(["id", "city", "country", "county", "countyFIPS","mostRecentPriceDomain","prices.currency", "prices.isSale"], axis=1, inplace=True)

    categorical_cols, numerical_cols = segregate_columns(df)

    for i in numerical_cols:
        df[i].fillna(df[i].mean(), inplace=True)
    for i in categorical_cols:
        df[i].fillna(df[i].mode()[0], inplace=True)

    df = drop_columns_with_high_null(df, threshold=0.75)

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

    # Save the trained model to a pickle file
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)


def predict_model(input_df: pd.DataFrame) -> pd.DataFrame:
    # Load the trained model from the pickle file
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # New dataframe for prediction
    input_df_cleaned = input_df.drop_duplicates()
    input_df_copy = input_df_cleaned.copy()
    input_df_cleaned = drop_columns_with_high_null(input_df_cleaned, threshold=0.75)
    input_df_cleaned.drop(["id", "city", "country", "county", "countyFIPS","mostRecentPriceDomain","prices.currency", "prices.isSale"], axis=1, inplace=True)

    categorical_cols, numerical_cols = segregate_columns(input_df_cleaned)

    for i in numerical_cols:
        input_df_cleaned[i].fillna(input_df_cleaned[i].mean(), inplace=True)
    for i in categorical_cols:
        input_df_cleaned[i].fillna(input_df_cleaned[i].mode()[0], inplace=True)

    input_df_cleaned = drop_columns_with_high_null(input_df_cleaned, threshold=0.75)

    for col in categorical_cols:
        label_encoder = LabelEncoder()
        input_df_cleaned[col] = label_encoder.fit_transform(input_df_cleaned[col])

    y_pred = model.predict(input_df_cleaned)

    input_df_copy['predicted_pricePerSquareFoot'] = y_pred

    return input_df_copy
