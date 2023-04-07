import mlflow

from .utils import resolve_outpath, rmpath


def log_figure(figure, filepath, drop_local=False, to_mlflow=True, mlflow_path=None):

    resolve_outpath(filepath)

    figure.savefig(filepath)

    if to_mlflow and (mlflow.active_run() is not None):

        mlflow.log_artifact(str(filepath), artifact_path=str(mlflow_path))

    if drop_local:

        rmpath(filepath)
