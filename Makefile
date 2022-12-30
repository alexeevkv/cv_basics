build_jupyter:
    python -m ipykernel install --user --name=venv
	jupyter contrib nbextension install --user
	jupyter nbextension enable varInspector/main
	jupyter serverextension enable --py jupyter_http_over_ws