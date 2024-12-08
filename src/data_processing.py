import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

def get_training_data():
    return  pd.read_csv('../data/train.csv'), pd.read_csv('../data/labels.csv')

def get_testing_data():
    return pd.read_csv('../data/test.csv')

def feature_encoding(X):
    """
    One-hot encode the 'features'.
    Input: X: features (pd.DataFrame)
    Output: X: features_encoded (pd.DataFrame)
    """
    non_numerical_columns_names = X.select_dtypes(exclude=['number']).columns
    le = LabelEncoder()
    for column in non_numerical_columns_names:
        # Only encore the columns that are not numerical 
        X[column] = le.fit_transform(X[column])

    return X

def normalize_features(X_train, X_validation, X_test, Is_Standard_scaler = True):
    """
    Take the input data and normalize the features.
    Input: X_train: features for train,  X_test: features for test (pd.DataFrame)
    Output: X_train_scaled, X_test_scaled (pd.DataFrame) the same shape of X_train and X_test
    """
    if Is_Standard_scaler: scaler = StandardScaler()
    else: scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_validation_scaled, X_test_scaled