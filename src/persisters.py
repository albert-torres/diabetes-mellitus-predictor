import os
import pickle
from datetime import date, datetime
from os import path

from src.settings import ROOT_DIR


class ModelPersister:
    location = f'{ROOT_DIR}/models'

    @classmethod
    def save(cls, model, description=None):
        directory_path = f'{cls.location}/{cls._get_directory_name()}'
        if not path.exists(directory_path):
            os.mkdir(directory_path)

        file_path = f'{directory_path}/{cls.get_model_name(model)}_{cls._get_current_time()}'
        if description is not None:
            file_path = f"{file_path}_{description}"

        with open(f"{file_path}.pickle", 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def get_model_name(model):
        return type(model).__name__

    @staticmethod
    def _get_directory_name():
        return date.today().strftime('%Y-%m-%d')

    @staticmethod
    def _get_current_time():
        return datetime.now().strftime('%H_%M_%S')
