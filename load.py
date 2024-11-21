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
    4. Adding custom features: 
       - Air temperature [K] * Process temperature [K]
       - Rotational speed [rpm] * Torque [Nm]
       - Torque [Nm] * Tool wear [min] * (Type == "L")
    5. Applying mean normalization to features.
    6. Splitting the dataset into features and labels.
    7. Splitting the dataset into training and testing sets.
    8. Balancing the dataset.
    """

    # Step 1: Load and clean data
    data = pd.read_csv(file_path)
    data_cleaned = data.dropna()

    # Step 2: Drop unnecessary columns
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    data_cleaned = data_cleaned.drop(columns=columns_to_drop, axis=1)

    # Step 3: Replace 'Type' column values
    pd.set_option('future.no_silent_downcasting', True)
    data_cleaned['Type'] = data_cleaned['Type'].replace({'L': 11, 'M': 12, 'H': 13}).infer_objects(copy=False).astype('float64')

    # Step 4: Add new features
    data_cleaned['AirTemp_ProcessTemp'] = data_cleaned['Air temperature [K]'] - data_cleaned['Process temperature [K]']
    data_cleaned['RotSpeed_Torque'] = data_cleaned['Rotational speed [rpm]'] * data_cleaned['Torque [Nm]']
    data_cleaned['Torque_ToolWear_TypeL'] = data_cleaned['Torque [Nm]'] * data_cleaned['Tool wear [min]'] * data_cleaned['Type']

    # Step 5: Drop the label column and normalize features
    data_dropped = data_cleaned.drop(['Machine failure'], axis=1)
    dropped = data_cleaned[['Machine failure']]
    data_normalized = mean_normalization(data_dropped)
    data_normalized = pd.concat([data_normalized, dropped], axis=1)

    # Step 6: Split into features and labels
    X, Y = split_x_and_y(data_normalized)

    # Step 7: Split into training and testing sets
    X_train, Y_train, X_test, Y_test = split_train_test(X, Y, train_ratio=0.7)

    # Step 8: Balance the dataset
    # X_train, Y_train = adasyn(X_train, Y_train, minority_class=1, beta=0.1, k=5)
    X_train, Y_train = cluster_undersample(X_train, Y_train, ratio=0.6)
    X_test, Y_test = cluster_undersample(X_test, Y_test, ratio=0.7)

    return X_train, Y_train, X_test, Y_test
