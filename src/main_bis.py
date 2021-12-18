import time
from statistics import mean

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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

train, test = train_test_split(measures_df, test_size=0.2, random_state=5, stratify=measures_df.diabetes)
train_x = train.loc[:, train.columns != 'diabetes']
train_x = StandardScaler().fit_transform(train_x)
test_x = test.loc[:, train.columns != 'diabetes']
test_x = StandardScaler().fit_transform(test_x)
train_y = train.diabetes
test_y = test.diabetes

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel='linear'),
    SVC(gamma=2),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier(),
    MLPClassifier(),
]

classifiers_scores = []

for model in classifiers:
    score = mean(cross_val_score(model, train_x, train_y))
    classifiers_scores.append(score)
    print(f"\nModel: {type(model).__name__}")
    print(f"Score: {score}")

# Best model
best_model_idx = classifiers_scores.index(max(classifiers_scores))
best_model = classifiers[best_model_idx].fit(train_x, train_y)
pred_y = best_model.predict(test_x)
print('\nBest model')
print(f'Model: {type(classifiers[best_model_idx]).__name__}')
print(f'Test accuracy score: {accuracy_score(test_y, pred_y)}')
print(f'Test AUC score: {roc_auc_score(test_y, pred_y)}')

# tree.plot_tree(dtc_model, proportion=True, ax=ax, fontsize=10, max_depth=3, filled=True, feature_names=c, class_names=["0", "1"])
# plt.savefig('decision_tree.jpeg')
# print(measures_diabetes_df.size)
# print(measures_diabetes_df[:50])
# print(measures_diabetes_df.size)
print(f'\nElapsed time {time.time() - start}')
