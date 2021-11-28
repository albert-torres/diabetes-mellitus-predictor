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
        with open(f'{ROOT_DIR}{path}') as f:
            lines = f.readlines()
        ids = []
        symptoms = []
        for line in lines[1:]:
            if len(splited_line := line.strip().split('\t')) < 2:
                continue
            ids.append(splited_line[0])
            symptoms.append(splited_line[1])

        symptomps_df = pd.DataFrame(list(zip(ids, symptoms)), columns=['ID', 'symptoms'])
        symptomps_df.drop_duplicates(inplace=True)
        return symptomps_df

