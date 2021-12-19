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

from src.persisters import DataPersister, ModelPersister
from src.readers import DatasetReader
from src.transformers import DataframeTransformer

pd.set_option('display.width', None)
pd.set_option('display.max_columns', 20)


def get_model_log_string(m, d=None):
    log_string = f'\nModel: {ModelPersister.get_model_name(m)}'
    if d:
        log_string = f'{log_string} ({d})'

    return log_string


start = time.time()

symptoms_df = DatasetReader.read_symptoms('/data/sintomas.csv')
extended_symptoms_df = DataframeTransformer.split_symptoms(symptoms_df)
print('Symptoms')
print(extended_symptoms_df.head())

diabetes_df = DatasetReader.read_data('/data/diabetes.csv')
measures_diabetes_df = DataframeTransformer.split_dataframe_first_measures(diabetes_df, 1)
measures_diabetes_df = DataframeTransformer.get_dummies(measures_diabetes_df, ['Sexo'])
measures_diabetes_df.dropna(inplace=True)
print(f'\nN diabetes (diabetes=1): {measures_diabetes_df.size}')
print(measures_diabetes_df.head())

no_diabetes_df = DatasetReader.read_data('/data/no_diabetes.csv')
measures_no_diabetes_df = DataframeTransformer.split_dataframe_last_measures(no_diabetes_df, 0)
measures_no_diabetes_df = DataframeTransformer.get_dummies(measures_no_diabetes_df, ['Sexo'])
measures_no_diabetes_df.dropna(inplace=True)
print(f'\nN no diabetes (diabetes=0): {measures_no_diabetes_df.size}')
print(measures_no_diabetes_df.head())

measures_df = pd.concat([measures_diabetes_df, measures_no_diabetes_df], ignore_index=True)
measures_df = DataframeTransformer.df_merge_left_on_column(measures_df, extended_symptoms_df, on='ID')
measures_df.fillna(0, inplace=True)
measures_df.drop(['ID'], axis='columns', inplace=True)
# Save processed data
DataPersister.save(measures_df, 'train_test_dataset_001.csv')

print(f'\nProcessed dataset: {measures_no_diabetes_df.size}')
print(measures_df.head())

# Training
train, test = train_test_split(measures_df, test_size=0.2, random_state=5, stratify=measures_df.diabetes)
train_x = train.loc[:, train.columns != 'diabetes']
train_x = StandardScaler().fit_transform(train_x)
test_x = test.loc[:, train.columns != 'diabetes']
test_x = StandardScaler().fit_transform(test_x)
train_y = train.diabetes
test_y = test.diabetes

classifiers = [
    (KNeighborsClassifier(), ''),
    (SVC(kernel='linear'), 'linear'),
    (SVC(gamma=2), 'gamma_2'),
    (DecisionTreeClassifier(), ''),
    (RandomForestClassifier(), ''),
    (AdaBoostClassifier(), ''),
    (GaussianNB(), ''),
    (QuadraticDiscriminantAnalysis(), ''),
    (SGDClassifier(), ''),
    (MLPClassifier(), ''),
]

classifiers_scores = []
for model, description in classifiers:
    # Save model
    ModelPersister.save(model, description)

    score = mean(cross_val_score(model, train_x, train_y))
    classifiers_scores.append(score)
    print(get_model_log_string(model, description))
    print(f'Score: {score}')

# Best model
best_model_idx = classifiers_scores.index(max(classifiers_scores))
best_model = classifiers[best_model_idx][0].fit(train_x, train_y)
pred_y = best_model.predict(test_x)
print('\nBest model')
print(get_model_log_string(classifiers[best_model_idx][0], classifiers[best_model_idx][1]))
print(f'Test accuracy score: {accuracy_score(test_y, pred_y)}')
print(f'Test AUC score: {roc_auc_score(test_y, pred_y)}')

print(f'\nElapsed time {time.time() - start}')
