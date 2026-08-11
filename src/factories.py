from typing import Optional

from src import datasets, metrics, models
from src.configs import DataConfigs, DecoderConfigs, ModelConfigs


def get_dataset(data_configs: DataConfigs, **kwargs):
    # Handle LAMP datasets - map LAMP_1, LAMP_2, etc. to LAMP class
    if data_configs.name.startswith('LAMP_'):
        dataset_class = getattr(datasets, 'LAMP')
    else:
        dataset_class = getattr(datasets, data_configs.name)
    
    return dataset_class(
        data_configs=data_configs,
        **kwargs,
    )


def get_model(
    model_configs: ModelConfigs,
    decoder_configs: DecoderConfigs,
):
    return getattr(models, decoder_configs.method)(
        model_configs=model_configs,
        decoder_configs=decoder_configs,
    )


def get_metrics(data_configs: DataConfigs):
    # Handle LAMP metrics - map LAMP_1, LAMP_2, etc. to LAMPMetric class
    if data_configs.name.startswith('LAMP_'):
        return getattr(metrics, 'LAMPMetric')()  # Instantiate the class
    else:
        return getattr(metrics, data_configs.name)
