import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score


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


def run_k_means(dataset, y):
    kmeans = KMeans(n_clusters=2, max_iter=2000)
    labels = kmeans.fit_predict(dataset)
    print(kmeans.inertia_)
    count = 0
    j = 0
    for i in labels:
        if i == y[j]:
            count += 1
        j += 1
    print(count)
    print(len(labels) - count)
    silhouette_avg = silhouette_score(X, labels)
    print (silhouette_avg)



if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    dataset = read_data("Churn_Modelling.csv")
    dataset = feature_scaling(dataset)
    X, y = remove_unwanted_features(dataset)
    X = label_encoder(X)
    X = hot_encoder(X)
    run_k_means(X,y)
