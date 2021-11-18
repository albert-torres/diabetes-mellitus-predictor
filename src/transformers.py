import numpy as np
import pandas as pd


class DataframeTransformer:
    @classmethod
    def create_dataframe_from_group(cls, identifier, grp, has_diabetes):
        data = {
            'ID': identifier,
            'Sexo': grp.Sexo.iloc[0],
            'Edad': grp.Edad.iloc[0],
            'Colesterol': (
                grp.loc[grp['Prueba'] == 'Colesterol'].Resultado.values[0] if
                len(grp.loc[grp['Prueba'] == 'Colesterol'].Resultado.values) != 0 else np.nan
            ),
            'LDL-Calculado': (
                grp.loc[grp['Prueba'] == 'LDL-Calculado'].Resultado.values[0] if
                len(grp.loc[grp['Prueba'] == 'LDL-Calculado'].Resultado.values) != 0 else np.nan
            ),
            'Hb-Glicosilada': (
                grp.loc[grp['Prueba'] == 'Hb-Glicosilada'].Resultado.values[0] if
                len(grp.loc[grp['Prueba'] == 'Hb-Glicosilada'].Resultado.values) != 0 else np.nan
            ),
            'Trigliceridos': (grp.loc[grp['Prueba'] == 'Trigliceridos'].Resultado.values[0] if
                              len(grp.loc[grp['Prueba'] == 'Trigliceridos'].Resultado.values) != 0 else np.nan),
            'HDL-Colesterol': (
                grp.loc[grp['Prueba'] == 'HDL-Colesterol'].Resultado.values[0] if
                len(grp.loc[grp['Prueba'] == 'HDL-Colesterol'].Resultado.values) != 0 else np.nan
            ),
            'diabetes': has_diabetes,
        }
        return pd.DataFrame(data, index=[0])

    @classmethod
    def split_dataframe(cls, df):
        grouped_by_id_date = df.groupby(['ID', 'Fecha'])
        measures_diabetes_df = pd.DataFrame()

        first_measure_id = None
        last_group = None
        last_id = list(grouped_by_id_date.groups)[0][0]

        for name, group in grouped_by_id_date:
            actual_id = name[0]
            if actual_id != first_measure_id:  # If actual_id is a new id then is the first measures
                df = cls.create_dataframe_from_group(actual_id, group, 1)
                measures_diabetes_df = measures_diabetes_df.append(df)
                first_measure_id = actual_id
            if actual_id != last_id:  # If actual_id is not last_id then the last_group is the last measures
                # Check if there is already the row with same values and diabetes=0. It is considered as first measure, thus,
                # diagnosed as diabetes, then -> stays as first measure.
                df = cls.create_dataframe_from_group(last_id, last_group, 0)
                duplicated = (
                    (measures_diabetes_df['ID'] == df['ID'][0]) &
                    (measures_diabetes_df['Sexo'] == df['Sexo'][0]) &
                    (measures_diabetes_df['Edad'] == df['Edad'][0]) &
                    (measures_diabetes_df['Colesterol'] == df['Colesterol'][0]) &
                    (measures_diabetes_df['LDL-Calculado'] == df['LDL-Calculado'][0]) &
                    (measures_diabetes_df['Hb-Glicosilada'] == df['Hb-Glicosilada'][0]) &
                    (measures_diabetes_df['Trigliceridos'] == df['Trigliceridos'][0]) &
                    (measures_diabetes_df['HDL-Colesterol'] == df['HDL-Colesterol'][0]) &
                    (measures_diabetes_df['diabetes'] == 1)
                ).any()
                if not duplicated:
                    measures_diabetes_df = measures_diabetes_df.append(df)
            last_id = actual_id
            last_group = group
        measures_diabetes_df.reset_index(drop=True, inplace=True)

        return measures_diabetes_df
