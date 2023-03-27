from pathlib import Path

import yaml
import json
import pickle
import pandas as pd

from .utils import resolve_outpath


def load_txt(filepath):
    with open(filepath, 'r') as f:
        return f.read()


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def dump_csv(obj, filepath):
    resolve_outpath(filepath)
    obj.to_csv(filepath, index=False)


def dump_txt(obj, filepath):
    resolve_outpath(filepath)
    with open(filepath, 'w') as f:
        f.write(obj)


def dump_json(obj, filepath):
    resolve_outpath(filepath)
    with open(filepath, 'wt') as f:
        json.dump(obj, f)


def dump_yaml(obj, filepath):
    resolve_outpath(filepath)
    with open(filepath, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)


def dump_pickle(obj, filepath):
    resolve_outpath(filepath)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_path_csv(path):
    path = Path(path)
    data = dict()

    for obj_path in path.iterdir():
        if obj_path.is_dir():
            data[obj_path.name] = load_path_csv(obj_path)
        elif obj_path.suffix == '.csv':
            data[obj_path.stem] = pd.read_csv(obj_path, index_col=0)

    return data
