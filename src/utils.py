from src.persisters import ModelPersister


def get_model_log_string(m, d=None):
    log_string = f'\nModel: {ModelPersister.get_model_name(m)}'
    if d:
        log_string = f'{log_string} ({d})'

    return log_string