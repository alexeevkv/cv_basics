from pathlib import Path
from copy import deepcopy
from functools import partial
import pathlib
from typing import Union, Dict, Any

import omegaconf
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate as hydra_instantiate


def instantiate(instantiate_obj: Dict[str, Any], *args, **kwargs) -> Any:
    """
    Function implementation from hydra. Additionally processes the key _partial_ 
    in instantinate object, wrapping the object in partial.
    https://hydra.cc/docs/advanced/instantiate_objects/overview/

    Parameters
    ----------
    instantiate_obj: dict

    Returns
    -------
    instantiate_obj: Any

    Notes
    -----
    Instantiate Object is a Dict with required field `_target_`,
    which specifies the import path of the Callable function.
    You can specify positional arguments in the field `_args_` field of this object.
    All other fields are used as kwargs of this function.

    Example of Instantiate Object:
    ```
    {
        '_target_': your.extractors.module.extractor,
        '_args_': ['positional', 'argument'],
        '_partial_': True,
        'param': 'value'
    }
    ```
    """
    if '_partial_' in instantiate_obj.keys():
        is_partial = instantiate_obj['_partial_']

        config_ = deepcopy(instantiate_obj)
        del config_['_partial_']

        if is_partial:
            return partial(hydra_instantiate, config_, *args, **kwargs)

    return hydra_instantiate(instantiate_obj, *args, **kwargs)


def resolve_config(dict_obj: dict) -> dict:
    """
    Implements pass through config and resolve instantinate objects in it.

    Parameters
    ----------
    dict_obj: dict
        Config with instantinate objects in it.

    Returns
    -------
    dict_obj: dict
        Dict with resolved instantinate objects.
    """
    if isinstance(dict_obj, dict):
        for key, value in dict_obj.items():
            if (isinstance(value, dict)) and ('_target_' in value):
                dict_obj[key] = instantiate(value)
            else:
                dict_obj[key] = resolve_config(value)
    return dict_obj


def prepare_config(config: Union[dict, DictConfig, str, pathlib.Path],
                   config_key: str = '',
                   resolve=False) -> Union[DictConfig, ListConfig]:
    """
    Implements the preparation of the config for its use in the script.

    Parameters
    ----------
    config: dict or DictConfig or str or pathlib.Path to config file
        Config with instantinate objects in it.

    config_key: str
        The key that will be loaded from the config. You can specify a nested key 
        by separating keys with a dot, for example `key1.key2.key3`.

    resolve: bool
        Pass through config and resolve instantinate objects in it or not.

    Returns
    -------
    config: DictConfig or ListConfig
        Prepared config.
    """
    if config_key:
        key_path = config_key.split('.')
    else:
        key_path = ''

    if isinstance(config, dict):
        config = OmegaConf.create(config)

    if (isinstance(config, str) or isinstance(config, Path)) and Path(config).exists:
        config = OmegaConf.load(config)
    else:
        raise Exception(f'Path `{config}` does not exist')
    
    if config_key not in config.keys():
        return None 

    if isinstance(config, omegaconf.dictconfig.DictConfig):
        for key in key_path:
            config = config[key]
    else:
        raise Exception(f'Unknown type of config `{type(config)}`')

    config = OmegaConf.to_object(config)
    
    if resolve:
        return resolve_config(config)

    return config
