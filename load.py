import pandas as pd
from preprocess import *

def mean_normalization(df):
    normalized_df = (df - df.mean()) / (df.max() - df.min())
    return normalized_df

def split_x_and_y(data):    
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    return features, labels

def split_train_test(features, labels, train_ratio=0.7):
    # shuffle the data
    indices = np.random.permutation(len(labels))
    features = features[indices]
    labels = labels[indices]
    # split the data
    train_size = int(train_ratio * len(labels))
    X_train = features[:train_size]
    Y_train = labels[:train_size]
    X_test = features[train_size:]
    Y_test = labels[train_size:]
    return X_train, Y_train, X_test, Y_test

    
def load_dataset(file_path):
    """
    1. Loading data and cleaning missing values.
    2. Dropping unnecessary columns.
    3. Replacing 'Type' column values: L -> -1, M -> 0, H -> 1.
    4. Applying mean normalization to features.
    5. Shuffling the dataset.
    6. Splitting the dataset into features and labels.
    7. Splitting the dataset into training and testing sets.
    """

    data = pd.read_csv(file_path)
    data_cleaned = data.dropna()
    
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    data_cleaned = data_cleaned.drop(columns=columns_to_drop, axis=1)
    
    pd.set_option('future.no_silent_downcasting', True)
    data_cleaned['Type'] = data_cleaned['Type'].replace({'L': -1, 'M': 0, 'H': 1}).infer_objects(copy=False).astype('float64')
    
    data_dropped = data_cleaned.drop(['Machine failure'], axis=1)
    dropped = data_cleaned[['Machine failure']]
    data_normalized = mean_normalization(data_dropped)
    data_normalized = pd.concat([data_normalized, dropped], axis=1)
    
    X, Y = split_x_and_y(data_normalized)
    X_train, Y_train, X_test, Y_test = split_train_test(X, Y, train_ratio=0.7)
    X_train, Y_train = adasyn(X_train, Y_train, 1, beta=0.9, k=5)
    
    return X_train, Y_train, X_test, Y_test