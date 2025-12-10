# Makefile - Project automation commands

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

train:
	python -m src.train

demo:
	streamlit run streamlit_app.py

verify:
	python verify_ann.py