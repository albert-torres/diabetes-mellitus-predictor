import os
import pickle
from abc import ABC
from datetime import date, datetime
from os import path

from src.settings import ROOT_DIR


class Persister(ABC):
    root_dir = f'{ROOT_DIR}'

    @staticmethod
    def _create_directory(directory_path):
        if not path.exists(directory_path):
            os.mkdir(directory_path)

    @staticmethod
    def _get_directory_name():
        return date.today().strftime('%Y-%m-%d')


class DataPersister(Persister):
    root_location = f'{Persister.root_dir}/processed_data'

    @classmethod
    def save(cls, df, filename):
        directory_path = f'{cls.root_location}/'
        cls._create_directory(directory_path)
        df.to_csv(f'{directory_path}/{filename}', index=False)


class ModelPersister(Persister):
    root_location = f'{Persister.root_dir}/models'

    @classmethod
    def save(cls, model, description=None):
        directory_path = f'{cls.root_location}/{cls._get_directory_name()}'
        cls._create_directory(directory_path)

        model_path = f'{directory_path}/{cls.get_model_name(model)}'
        if description:
            model_path = f'{model_path}_{description}'

        with open(f'{model_path}_{cls._get_current_time()}.pickle', 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def get_model_name(model):
        return type(model).__name__

    @staticmethod
    def _get_current_time():
        return datetime.now().strftime('%H_%M_%S')
