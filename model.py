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


def remove_outliers_iqr(df, columns, threshold=1.5):
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)

    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Define the upper and lower bounds for outlier detection
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Remove rows containing outliers
    filtered_df = df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]

    return filtered_df

def select_10_best_features(X,y,df):
    target_column = 'prices.pricePerSquareFoot'
    selector = SelectKBest(f_regression, k=10)  # Select top 10 features based on f_regression score
    X_selected = selector.fit_transform(X, y)
    selected_feature_indices = selector.get_support(indices=True)
    selected_features = df.drop(target_column, axis=1).columns[selected_feature_indices]
    return selected_features
def y_outliers_removal(df):
    Q1 = df['prices.pricePerSquareFoot'].quantile(0.25)
    Q3 = df['prices.pricePerSquareFoot'].quantile(0.75)
    threshold=1.3
    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Define the upper and lower bounds for outlier detection
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers_low=(df['prices.pricePerSquareFoot']<lower_bound)
    outliers_high=(df['prices.pricePerSquareFoot']>upper_bound)
    df['prices.pricePerSquareFoot'][outliers_low|outliers_high]
    df=df[~(outliers_low|outliers_high)]
    return df

def train_model():
    df = pd.read_csv('compressed data.csv')

    df = df.drop_duplicates()
    df = drop_columns_with_high_null(df, threshold=0.75)
    df.drop(["Unnamed: 0","id", "city", "country", "county", "countyFIPS","mostRecentPriceDomain","prices.currency", "prices.isSale"], axis=1, inplace=True)

    categorical_cols, numerical_cols = segregate_columns(df)

    for i in numerical_cols:
        df[i].fillna(df[i].mean(), inplace=True)
    for i in categorical_cols:
        df[i].fillna(df[i].mode()[0], inplace=True)


    for col in categorical_cols:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])
        
    df=y_outliers_removal(df)
    target_column = 'prices.pricePerSquareFoot'
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    
    #selector = SelectKBest(f_regression, k=10)  # Select top 10 features based on f_regression score
    #X_selected = selector.fit_transform(X, y)
    #selected_feature_indices = selector.get_support(indices=True)

# Get the names of the selected features
    #selected_features = df.drop(target_column, axis=1).columns[selected_feature_indices]
    #print(selected_features)
    #print(select_10_best_features(X,y))
    #columns_to_check = selected_features
    #df_without_outliers = remove_outliers_iqr(df, columns_to_check, threshold=1.5)
    
    #print("Top 10 Features:")
    #for feature in selected_features:
     #   print(feature)
 
   # Assuming your DataFrame is named 'df'
   # Replace 'columns_to_check' with the columns you want to check for outliers
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(X)  # Drop the target column before scaling

    # Step 4: Feature Selection

    X = df_scaled
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X,y)
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
    input_df_cleaned.drop(['apiURLs', 'buildingName', 'congressionalDistrictHouse', 'deposits', 'languagesSpoken', 'leasingTerms', 'numPeople', 'numUnit', 'paymentTypes', 'people', 'petPolicy', 'prices.comment', 'prices.date', 'prices.dateValidStart', 'prices.dateValidEnd', 'prices.minStay', 'prices.period', 'reviews', 'rules', 'taxID', 'yearBuilt', 'taxID.1'],axis=1,inplace=True)
    input_df_cleaned.drop(["Unnamed: 0.1","Unnamed: 0","id", "city", "country", "county", "countyFIPS","mostRecentPriceDomain","prices.currency", "prices.isSale"], axis=1, inplace=True)

    categorical_cols, numerical_cols = segregate_columns(input_df_cleaned)

    for i in numerical_cols:
        input_df_cleaned[i].fillna(input_df_cleaned[i].mean(), inplace=True)
    for i in categorical_cols:
        input_df_cleaned[i].fillna(input_df_cleaned[i].mode()[0], inplace=True)

    for col in categorical_cols:
        label_encoder = LabelEncoder()
        input_df_cleaned[col] = label_encoder.fit_transform(input_df_cleaned[col])

    y_pred = model.predict(input_df_cleaned)

    input_df_copy['predicted_pricePerSquareFoot'] = y_pred

    return input_df_copy
