import time
from statistics import mean

import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from src.readers import DatasetReader
from src.transformers import DataframeTransformer

pd.set_option('display.width', None)
pd.set_option('display.max_columns', 10)

start = time.time()

diabetes_df = DatasetReader.read_data('/data/diabetes.csv')
no_diabetes_df = DatasetReader.read_data('/data/no_diabetes.csv')
symptoms_df = DatasetReader.read_symptoms('/data/sintomas.csv')

# Grouping (Erase groups that have not all Pruebas done?)
measures_diabetes_df = DataframeTransformer.split_dataframe(diabetes_df)
print(measures_diabetes_df.size)
measures_diabetes_df.dropna(inplace=True)
measures_diabetes_df.drop(['ID', 'Sexo'], axis='columns', inplace=True)
print(measures_diabetes_df.size)

train, test = train_test_split(measures_diabetes_df, test_size=0.2, random_state=5)
knn_model = KNeighborsClassifier(n_neighbors=8)
gnb = GaussianNB()
scores = cross_val_score(knn_model, train.loc[:, train.columns != 'diabetes'], train.diabetes)
print(mean(scores))

# from apyori import apriori
#
# associations = apriori(measures_diabetes_df, min_length=2, min_support=0.2, min_confidence=0.2, min_lift=3)
# associations = list(associations)
# print(associations[0])
# print(associations)

# print(measures_diabetes_df.size)
# print(measures_diabetes_df[:50])
# print(measures_diabetes_df.size)
end = time.time()
print(end - start)
