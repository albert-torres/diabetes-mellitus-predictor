import pandas as pd

from src.settings import ROOT_DIR


class DatasetReader:
    @staticmethod
    def read_data(path):
        df = pd.read_csv(f'{ROOT_DIR}{path}')
        df['Fecha'] = pd.to_datetime(df.Fecha, format='%d/%m/%Y')
        df.drop_duplicates(inplace=True)
        df.sort_values(['ID', 'Fecha'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def read_symptoms(path):
        df = pd.read_csv(f'{ROOT_DIR}{path}')
        symptoms_column_names = [f'symptom_{n + 1}' for n in range(df.shape[1] - 1)]
        df.columns = ['ID'] + symptoms_column_names
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df[symptoms_column_names] = df[symptoms_column_names].applymap(
            lambda x: x.strip().lower() if type(x) == str else x
        )

        return df
