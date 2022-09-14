import os
import glob
import math
import pandas as pd
import numpy as np
import timeit
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB

start = timeit.default_timer()
missing_dataset = pd.read_csv(
    "/Users/shivani/Desktop/SHIVU/DM project/Incomplete datasets(1)/Data 9/Data_9_NW_20%.csv", header=None)
learning_dataset = missing_dataset.dropna()
ImputedDataset = missing_dataset
original = pd.read_csv(
    '/Users/shivani/Desktop/SHIVU/DM project/Complete datasets(1)/Data_9.csv')

model = GaussianNB()
d = defaultdict(LabelEncoder)
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
ImputedDataset = imp.fit_transform(ImputedDataset)

learning_dataset = learning_dataset.apply(lambda x: d[x.name].fit_transform(x))
ImputedDataset = pd.DataFrame(ImputedDataset).apply(
    lambda x: d[x.name].fit_transform(x))

for i in range(len(missing_dataset.columns)-1):
    X_Train_df = learning_dataset.iloc[:, learning_dataset.columns != i]
    y_Train_df = learning_dataset.iloc[:, lambda dataset: [i]]
    X_Train = X_Train_df.values
    y_Train = y_Train_df.values

    model.fit(X_Train, y_Train.ravel())

    X_temp = pd.DataFrame(ImputedDataset)
    X_temp = X_temp.iloc[:, X_temp.columns != i]

    y_imp = model.predict(X_temp)
    y_impdf = pd.DataFrame(y_imp)

    ImputedDataset_df = pd.DataFrame(ImputedDataset)
    for j in range(len(ImputedDataset_df.index)):
        if pd.isna(missing_dataset.iloc[j, i]):
            ImputedDataset[i][j] = y_imp[j]
            ImputedDataset_df = pd.DataFrame(ImputedDataset)

    del y_impdf
    del y_Train_df
    del X_Train_df
    del X_temp
    del y_imp
    del X_Train
    del y_Train

Imputed_Dataset = ImputedDataset_df.apply(
    lambda x: d[x.name].inverse_transform(x))
end = timeit.default_timer()
total_time = end-start

difference_sum = 0.0
total = 0.0

for i in range(len(original.index)):
    for j in range(len(original.columns)-1):
        difference = float(
            Imputed_Dataset.iloc[i, j]-float(original.iloc[i, j]))
        difference_square = difference*difference
        difference_sum = difference_sum+difference_square
        total = total+(float(original.iloc[i, j])*float(original.iloc[i, j]))

NRMS = math.sqrt(difference_sum)/math.sqrt(total)
print("Time: ", total_time)
print("NRMS: ", NRMS)