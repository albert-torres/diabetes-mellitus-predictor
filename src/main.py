from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve(strict=True).parent.parent

diabetes_df = pd.read_csv(f'{ROOT_DIR}/data/diabetes.csv')
no_diabetes_df = pd.read_csv(f'{ROOT_DIR}/data/no_diabetes.csv')

with open(f'{ROOT_DIR}/data/sintomas.csv') as f:
    lines = f.readlines()
ids = []
symptoms = []
for line in lines[1:]:
    if len(splited_line := line.strip().split('\t')) < 2:
        continue
    ids.append(splited_line[0])
    symptoms.append(splited_line[1])

symptoms_df = pd.DataFrame(list(zip(ids, symptoms)), columns=['ID', 'symptoms'])

# print(diabetes_df.describe())
# print(no_diabetes_df.describe())
# print(symptoms_df.describe())

diabetes_df['Fecha'] = pd.to_datetime(diabetes_df.Fecha, format='%d/%m/%Y')
diabetes_df.sort_values(['ID', 'Fecha'], inplace=True)
diabetes_df.reset_index(drop=True, inplace=True)

no_diabetes_df['Fecha'] = pd.to_datetime(no_diabetes_df.Fecha, format='%d/%m/%Y')
no_diabetes_df.sort_values(['ID', 'Fecha'], inplace=True)
no_diabetes_df.reset_index(drop=True, inplace=True)


