import shutil
from typing import Union
from pathlib import Path


def rmpath(path: Union[str, Path]):
    """
    Remove directory or file at the specified path.

    Parameters
    ----------
    path: str or pathlib.Path
    """
    if Path(path).is_dir():
        shutil.rmtree(path)
    if Path(path).is_file():
        Path(path).unlink()


def resolve_outpath(path: Union[str, Path]):
    """
    Creates a directory at the specified path if it does not exist.

    Parameters
    ----------
    path: str or pathlib.Path
    """
    path = Path(path).resolve()

    # If there is no suffix, then we consider that the path
    # is a directory, otherwise resolve parent as path
    if path.suffix != '':
        path = path.parent

    Path.mkdir(path, parents=True, exist_ok=True)
