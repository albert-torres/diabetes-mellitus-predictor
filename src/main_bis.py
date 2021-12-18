import time
from collections import Counter
from statistics import mean

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.readers import DatasetReader
from src.transformers import DataframeTransformer

pd.set_option('display.width', None)
pd.set_option('display.max_columns', 20)

start = time.time()

diabetes_df = DatasetReader.read_data('/data/diabetes.csv')
no_diabetes_df = DatasetReader.read_data('/data/no_diabetes.csv')
symptoms_df = DatasetReader.read_symptoms('/data/sintomas.csv')

# Grouping (Erase groups that have not all Pruebas done?)
measures_diabetes_df = DataframeTransformer.split_dataframe_first_measures(diabetes_df, 1)
measures_diabetes_df = DataframeTransformer.merge_dfs_on_column(measures_diabetes_df, symptoms_df, on='ID')
measures_diabetes_df = measures_diabetes_df[
    (measures_diabetes_df['symptoms'] == 'Diabetes') |
    (measures_diabetes_df['symptoms'] == 'Diabetes, sobrepeso')
]
measures_diabetes_df.drop('symptoms', axis='columns', inplace=True)
measures_diabetes_df = DataframeTransformer.get_dummies(measures_diabetes_df, ['Sexo'])
measures_diabetes_df.dropna(inplace=True)
print(f"N diabetes (diabetes=1): {measures_diabetes_df.size}")
print(measures_diabetes_df.head())

measures_no_diabetes_df = DataframeTransformer.split_dataframe_first_measures(no_diabetes_df, 0)
# measures_no_diabetes_df = DataframeTransformer.merge_dfs_on_column(measures_no_diabetes_df, symptoms_df, on='ID')
measures_no_diabetes_df = DataframeTransformer.get_dummies(measures_no_diabetes_df, ['Sexo'])
measures_no_diabetes_df.dropna(inplace=True)
print(f"N no diabetes (diabetes=0): {measures_no_diabetes_df.size}")
print(measures_no_diabetes_df.head())

measures_df = pd.concat([measures_diabetes_df, measures_no_diabetes_df], ignore_index=True)
measures_df.fillna(0, inplace=True)
measures_df.drop(['ID'], axis='columns', inplace=True)
# print(measures_df.size)
# print(measures_df.head())

# Training

train, test = train_test_split(measures_df, test_size=0.2, random_state=5, stratify=measures_df.diabetes)
train_x = train.loc[:, train.columns != 'diabetes']
test_x = test.loc[:, train.columns != 'diabetes']
train_y = train.diabetes
test_y = test.diabetes

knn_model = KNeighborsClassifier(n_neighbors=3)
gnb_model = GaussianNB()
svc_model = SVC()
stochastic_model = SGDClassifier()
dtc_model = DecisionTreeClassifier()
models = [knn_model, gnb_model, svc_model, stochastic_model, dtc_model]
for model in models:
    scores = cross_val_score(model, train_x, train_y)
    print(f"\nModel: {type(model).__name__}")
    print(f"Score: {mean(scores)}")

dtc_model = dtc_model.fit(train_x, train_y)

# Test

pred_y = dtc_model.predict(test_x)
print(f'Test score: {accuracy_score(test_y, pred_y)}')

# tree.plot_tree(dtc_model, proportion=True, ax=ax, fontsize=10, max_depth=3, filled=True, feature_names=c, class_names=["0", "1"])
# plt.savefig('decision_tree.jpeg')
# print(measures_diabetes_df.size)
# print(measures_diabetes_df[:50])
# print(measures_diabetes_df.size)
end = time.time()
print(end - start)
