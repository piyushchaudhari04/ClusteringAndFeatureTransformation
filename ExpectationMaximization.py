import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def read_data(file_name):
    return pd.read_csv(file_name)


def remove_unwanted_features(dataset):
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values
    return X, y


def label_encoder(X):
    label_encoder_X = LabelEncoder()
    X[:, 1] = label_encoder_X.fit_transform(X[:, 1])
    X[:, 2] = label_encoder_X.fit_transform(X[:, 2])
    return X


def hot_encoder(X):
    one_hot_encoder = OneHotEncoder(categorical_features=[1])
    X = one_hot_encoder.fit_transform(X).toarray()
    return X


def feature_scaling(dataset):
    scaler = StandardScaler()
    cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
    dataset[cols] = scaler.fit_transform(dataset[cols])
    return dataset


def run_expectation_maximization(dataset, y):
    gmm = GaussianMixture(n_components=2, random_state=10)
    labels = gmm.fit_predict(X)
    expected_labels = y
    j = 0
    count = 0
    for i in labels:
        if i == expected_labels[j]:
            count += 1
        j += 1

    print(count)
    print(len(labels) - count)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    dataset = read_data("Churn_Modelling.csv")
    dataset = feature_scaling(dataset)
    X, y = remove_unwanted_features(dataset)
    X = label_encoder(X)
    X = hot_encoder(X)
    run_expectation_maximization(X, y)
